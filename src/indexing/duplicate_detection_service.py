"""
Deduplication Service backed by Supabase
Handles exact file hash match (Layer 1), near-duplicate MinHash/pHash (Layer 2), and data-level match (Layer 3).
"""
import hashlib
import io
from datasketch import MinHash
from src.config.config import HUMMING_DISTANCE, JACCARED_SIMILARITY
from src.utils.supabase_client import (
    check_exact_duplicate,
    get_all_document_hashes,
    insert_document_hash,
    check_data_level_duplicate
)

class DuplicateDetectionService:
    def __init__(self):
        pass

    def generate_sha256_hash(self, file_content: bytes) -> str:
        """Generates a SHA-256 hash for the raw file contents."""
        sha256_hash = hashlib.sha256()
        sha256_hash.update(file_content)
        return sha256_hash.hexdigest()

    def generate_minhash(self, text: str) -> str:
        """Generates a MinHash signature string for near-duplicate text detection."""
        if not text:
            return ""
        m = MinHash(num_perm=128)
        for word in text.split():
            m.update(word.encode('utf8'))
        return ",".join(map(str, m.hashvalues))

    def generate_phash(self, file_content: bytes, filename: str) -> str:
        """Generates a Perceptual Hash (pHash) for image files."""
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return ""
        try:
            import imagehash
            from PIL import Image
            image = Image.open(io.BytesIO(file_content))
            return str(imagehash.phash(image))
        except Exception as e:
            print(f"[-] Failed to generate pHash: {e}")
            return ""

    def is_exact_duplicate(self, file_content: bytes) -> bool:
        """Checks Supabase 'document_hashes' table (Layer 1) for exact binary match."""
        file_hash = self.generate_sha256_hash(file_content)
        return check_exact_duplicate(file_hash)

    def is_near_duplicate(self, text: str, file_content: bytes = None, filename: str = "") -> bool:
        """Checks Supabase 'document_hashes' table (Layer 2) for MinHash or pHash collisions."""
        min_hash = self.generate_minhash(text) if text else ""
        p_hash = self.generate_phash(file_content, filename) if file_content else ""
        
        if not min_hash and not p_hash:
            return False
            
        try:
            entities = get_all_document_hashes()
            
            incoming_minhash = None
            if text:
                incoming_minhash = MinHash(num_perm=128)
                for word in text.split():
                    incoming_minhash.update(word.encode('utf8'))
                    
            incoming_phash = None
            if file_content and p_hash:
                import imagehash
                incoming_phash = imagehash.hex_to_hash(p_hash)
                
            for entity in entities:
                # 1. Visual Similarity Check via Hamming Distance
                if incoming_phash and entity.get("phash_signature"):
                    try:
                        import imagehash
                        existing_phash = imagehash.hex_to_hash(entity["phash_signature"])
                        if incoming_phash - existing_phash < HUMMING_DISTANCE:
                            print("[DUPE-DETECT] pHash match found! Images are visually identical.")
                            return True
                    except Exception:
                        pass
                        
                # 2. Text Similarity Check via Jaccard Index
                if incoming_minhash and entity.get("minhash_signature"):
                    try:
                        sig_str = entity["minhash_signature"]
                        if sig_str:
                            existing_hashvalues = list(map(int, sig_str.split(",")))
                            existing_minhash = MinHash(num_perm=128)
                            existing_minhash.hashvalues = existing_hashvalues
                            
                            similarity = incoming_minhash.jaccard(existing_minhash)
                            if similarity >= JACCARED_SIMILARITY:
                                print(f"[DUPE-DETECT] MinHash match found! Similarity: {similarity*100:.2f}%")
                                return True
                    except Exception:
                        pass
                        
            return False
        except Exception as e:
            print(f"[-] Failed to calculate near duplicates: {e}")
            return False

    def is_data_level_duplicate(self, metadata: dict) -> bool:
        """Checks Supabase 'clinical_records' (Layer 3) for matching document_number & entity_name."""
        doc_num = metadata.get("document_number")
        entity_name = metadata.get("entity_name") or metadata.get("patient_name")
        
        if not doc_num or doc_num in ["N/A", "null", None] or not entity_name or entity_name in ["N/A", "null", None]:
            return False
            
        return check_data_level_duplicate(doc_num, entity_name)

    def log_document(self, file_content: bytes, document_id: str, text: str = "", filename: str = ""):
        """Logs document hashes into Supabase 'document_hashes' table."""
        file_hash = self.generate_sha256_hash(file_content)
        min_hash = self.generate_minhash(text)
        p_hash = self.generate_phash(file_content, filename)
        insert_document_hash(file_hash, document_id, min_hash, p_hash)
