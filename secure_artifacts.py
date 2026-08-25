"""Encrypt or decrypt local data artifacts (SQLite + vector index directory).

Usage examples:
    $env:DATA_ENCRYPTION_PASSPHRASE="YOUR_LONG_RANDOM_PASSPHRASE"
    uv run python secure_artifacts.py encrypt --passphrase-env DATA_ENCRYPTION_PASSPHRASE
    uv run python secure_artifacts.py decrypt --passphrase-env DATA_ENCRYPTION_PASSPHRASE
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import tarfile
from pathlib import Path

from dotenv import load_dotenv
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from settings import (
    CAMPAIGN_DB_ENC_FILE_NAME,
    CAMPAIGN_DB_FILE_NAME,
    ENCRYPTION_DEFAULT_PASSPHRASE_ENV,
    ENCRYPTION_MAGIC,
    ENCRYPTION_NONCE_SIZE,
    ENCRYPTION_PBKDF2_ITERATIONS,
    ENCRYPTION_SALT_SIZE,
    ROOT,
    VECTOR_DB_DIR_NAME,
    VECTOR_DB_ENC_FILE_NAME,
)

MAGIC = ENCRYPTION_MAGIC
SALT_SIZE = ENCRYPTION_SALT_SIZE
NONCE_SIZE = ENCRYPTION_NONCE_SIZE
PBKDF2_ITERATIONS = ENCRYPTION_PBKDF2_ITERATIONS

DEFAULT_SQLITE = ROOT / CAMPAIGN_DB_FILE_NAME
DEFAULT_VECTOR_DIR = ROOT / VECTOR_DB_DIR_NAME
DEFAULT_SQLITE_ENC = ROOT / CAMPAIGN_DB_ENC_FILE_NAME
DEFAULT_VECTOR_ENC = ROOT / VECTOR_DB_ENC_FILE_NAME

# Allow CLI and runtime helpers to pick up passphrases from repository root .env.
load_dotenv(ROOT / ".env")


def derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 256-bit key from a passphrase and random salt."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=PBKDF2_ITERATIONS,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_bytes(plaintext: bytes, passphrase: str) -> bytes:
    """Encrypt bytes and return a self-contained payload."""
    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = derive_key(passphrase, salt)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return MAGIC + salt + nonce + ciphertext


def decrypt_bytes(payload: bytes, passphrase: str) -> bytes:
    """Decrypt a self-contained payload created by encrypt_bytes."""
    if not payload.startswith(MAGIC):
        raise ValueError("Unsupported encrypted payload format.")
    min_size = len(MAGIC) + SALT_SIZE + NONCE_SIZE + 16
    if len(payload) < min_size:
        raise ValueError("Encrypted payload is incomplete.")
    offset = len(MAGIC)
    salt = payload[offset:offset + SALT_SIZE]
    offset += SALT_SIZE
    nonce = payload[offset:offset + NONCE_SIZE]
    ciphertext = payload[offset + NONCE_SIZE:]
    key = derive_key(passphrase, salt)
    return AESGCM(key).decrypt(nonce, ciphertext, None)


def encrypt_file(input_path: Path, output_path: Path, passphrase: str) -> None:
    plaintext = input_path.read_bytes()
    output_path.write_bytes(encrypt_bytes(plaintext, passphrase))


def decrypt_file(input_path: Path, output_path: Path, passphrase: str) -> None:
    payload = input_path.read_bytes()
    output_path.write_bytes(decrypt_bytes(payload, passphrase))


def pack_directory(directory: Path) -> bytes:
    """Create a compressed tar payload for one directory."""
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        archive.add(directory, arcname=directory.name)
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb") as gz_file:
        gz_file.write(tar_buffer.getvalue())
    return compressed.getvalue()


def unpack_directory(payload: bytes, target_parent: Path) -> Path:
    """Extract a gzip-tar payload into target_parent and return extracted root."""
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as gz_file:
        tar_data = gz_file.read()
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r") as archive:
        base = target_parent.resolve()
        for member in archive.getmembers():
            destination = (target_parent / member.name).resolve()
            if not str(destination).startswith(str(base)):
                raise ValueError("Refusing to extract archive with unsafe paths.")
        root_names = [member.name.split("/")[0] for member in archive.getmembers() if member.name]
        root_name = root_names[0] if root_names else "vector_db"
        try:
            archive.extractall(path=target_parent, filter="data")
        except TypeError:
            archive.extractall(path=target_parent)
    return target_parent / root_name


def get_passphrase(env_name: str) -> str:
    passphrase = os.getenv(env_name, "")
    if not passphrase:
        raise ValueError(
            f"Missing passphrase in environment variable '{env_name}'. "
            "Set it in your shell before running this command."
        )
    return passphrase


def run_encrypt(args: argparse.Namespace) -> None:
    passphrase = get_passphrase(args.passphrase_env)
    sqlite_path = args.sqlite.resolve()
    vector_dir = args.vector_dir.resolve()
    sqlite_out = args.sqlite_out.resolve()
    vector_out = args.vector_out.resolve()

    if not sqlite_path.exists():
        raise FileNotFoundError(sqlite_path)
    if not vector_dir.exists() or not vector_dir.is_dir():
        raise FileNotFoundError(vector_dir)

    encrypt_file(sqlite_path, sqlite_out, passphrase)
    vector_payload = pack_directory(vector_dir)
    vector_out.write_bytes(encrypt_bytes(vector_payload, passphrase))

    if args.remove_plain:
        sqlite_path.unlink()
        for child in sorted(vector_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        vector_dir.rmdir()

    print(f"Encrypted SQLite -> {sqlite_out}")
    print(f"Encrypted vector DB -> {vector_out}")


def run_decrypt(args: argparse.Namespace) -> None:
    passphrase = get_passphrase(args.passphrase_env)
    sqlite_enc = args.sqlite_enc.resolve()
    vector_enc = args.vector_enc.resolve()
    sqlite_out = args.sqlite_out.resolve()
    vector_parent = args.vector_parent.resolve()

    if not sqlite_enc.exists():
        raise FileNotFoundError(sqlite_enc)
    if not vector_enc.exists():
        raise FileNotFoundError(vector_enc)

    decrypt_file(sqlite_enc, sqlite_out, passphrase)
    vector_payload = decrypt_bytes(vector_enc.read_bytes(), passphrase)
    extracted = unpack_directory(vector_payload, vector_parent)

    print(f"Decrypted SQLite -> {sqlite_out}")
    print(f"Decrypted vector DB -> {extracted}")


def decrypt_artifacts_for_runtime(
    passphrase: str,
    sqlite_enc: Path = DEFAULT_SQLITE_ENC,
    vector_enc: Path = DEFAULT_VECTOR_ENC,
    sqlite_out: Path = DEFAULT_SQLITE,
    vector_parent: Path = ROOT,
) -> tuple[Path, Path]:
    """Ensure encrypted artifacts are available as plaintext for runtime use."""
    sqlite_target = sqlite_out.resolve()
    vector_root = (vector_parent / VECTOR_DB_DIR_NAME).resolve()

    if not sqlite_target.exists():
        if not sqlite_enc.exists():
            raise FileNotFoundError(f"Missing encrypted SQLite file: {sqlite_enc}")
        decrypt_file(sqlite_enc.resolve(), sqlite_target, passphrase)

    if not vector_root.exists():
        if not vector_enc.exists():
            raise FileNotFoundError(f"Missing encrypted vector archive: {vector_enc}")
        vector_payload = decrypt_bytes(vector_enc.resolve().read_bytes(), passphrase)
        unpack_directory(vector_payload, vector_parent.resolve())

    return sqlite_target, vector_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encrypt/decrypt local campaign artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encrypt = subparsers.add_parser("encrypt", help="Encrypt sqlite and vector_db artifacts.")
    encrypt.add_argument("--passphrase-env", default=ENCRYPTION_DEFAULT_PASSPHRASE_ENV)
    encrypt.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    encrypt.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    encrypt.add_argument("--sqlite-out", type=Path, default=DEFAULT_SQLITE_ENC)
    encrypt.add_argument("--vector-out", type=Path, default=DEFAULT_VECTOR_ENC)
    encrypt.add_argument("--remove-plain", action="store_true", help="Delete unencrypted artifacts after encrypting.")
    encrypt.set_defaults(func=run_encrypt)

    decrypt = subparsers.add_parser("decrypt", help="Decrypt sqlite and vector_db artifacts.")
    decrypt.add_argument("--passphrase-env", default=ENCRYPTION_DEFAULT_PASSPHRASE_ENV)
    decrypt.add_argument("--sqlite-enc", type=Path, default=DEFAULT_SQLITE_ENC)
    decrypt.add_argument("--vector-enc", type=Path, default=DEFAULT_VECTOR_ENC)
    decrypt.add_argument("--sqlite-out", type=Path, default=DEFAULT_SQLITE)
    decrypt.add_argument("--vector-parent", type=Path, default=ROOT)
    decrypt.set_defaults(func=run_decrypt)

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


