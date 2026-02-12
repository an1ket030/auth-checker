from passlib.context import CryptContext
try:
    ctx = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
    h = ctx.hash("password123")
    print(f"HASH_GENERATED: {h}")
    print(f"VERIFY_RESULT: {ctx.verify('password123', h)}")
except Exception as e:
    print(f"ERROR: {e}")
