import hmac
import hashlib

def generate_hash(server_seed, client_seed, nonce):
    msg = f"{client_seed}:{nonce}".encode()
    return hmac.new(server_seed.encode(), msg, hashlib.sha256).hexdigest()
