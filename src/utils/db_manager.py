import json
import shutil
import os
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHROMA_PATH = "/users/hkr/library/mobile documents/com~apple~clouddocs/ai/claude-memory/chroma_db"
BACKUP_PATH = "/users/hkr/library/mobile documents/com~apple~clouddocs/ai/claude-memory/backups"

class DatabaseManager:
    def __init__(self, chroma_client):
        self.client = chroma_client
        try:
            self.collection = self.client.get_collection("memory_collection")
            logger.info("Found existing collection in DatabaseManager")
        except:
            logger.info("Creating new collection in DatabaseManager")
            self.collection = self.client.create_collection(
                name="memory_collection",
                metadata={"hnsw:space": "cosine"}
            )
        os.makedirs(BACKUP_PATH, exist_ok=True)

    def _reinitialize_collection(self):
        """Reinitialize the collection reference"""
        try:
            self.collection = self.client.get_collection("memory_collection")
            logger.info("Successfully reinitialized collection reference")
            return True
        except Exception as e:
            logger.error(f"Failed to reinitialize collection: {str(e)}")
            return False

    def create_backup(self):
        """Create a backup of the current ChromaDB"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        attempt = 0
        while True:
            backup_dir = os.path.join(BACKUP_PATH, f"backup_{timestamp}")
            if attempt > 0:
                backup_dir = f"{backup_dir}_{attempt}"
            if not os.path.exists(backup_dir):
                break
            attempt += 1
        
        try:
            # Create backup of ChromaDB files
            shutil.copytree(CHROMA_PATH, backup_dir)
            
            # Export metadata
            all_data = self.collection.get()
            metadata = {
                'timestamp': timestamp,
                'total_memories': len(all_data['ids']),
                'data': all_data
            }
            
            with open(os.path.join(backup_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f)
                
            logger.info(f"Backup created at: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            logger.error(f"Backup failed: {str(e)}")
            raise

    def restore_backup(self, backup_dir):
        """Restore from a specific backup"""
        try:
            # Verify backup exists
            if not os.path.exists(backup_dir):
                raise ValueError(f"Backup directory not found: {backup_dir}")
                
            # Restore files
            shutil.rmtree(CHROMA_PATH, ignore_errors=True)
            shutil.copytree(backup_dir, CHROMA_PATH)
            
            # Reinitialize the collection reference
            if not self._reinitialize_collection():
                raise Exception("Failed to reinitialize collection after restore")
            
            logger.info(f"Restored from backup: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            raise

    def cleanup_old_backups(self, max_age_days=30):
        """Remove backups older than specified days"""
        try:
            current_time = time.time()
            count = 0
            
            for backup_dir in os.listdir(BACKUP_PATH):
                backup_path = os.path.join(BACKUP_PATH, backup_dir)
                if os.path.isdir(backup_path):
                    # Check directory age
                    dir_time = os.path.getctime(backup_path)
                    age_days = (current_time - dir_time) / (24 * 3600)
                    
                    if age_days > max_age_days:
                        shutil.rmtree(backup_path)
                        count += 1
                        
            logger.info(f"Removed {count} old backups")
            return count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise

    def get_memory_stats(self):
        """Get detailed statistics about stored memories"""
        try:
            all_data = self.collection.get()
            if not all_data or not all_data.get('ids'):
                return {
                    'total_memories': 0,
                    'tags': {},
                    'types': {},
                    'time_distribution': {
                        'last_24h': 0,
                        'last_week': 0,
                        'last_month': 0,
                        'older': 0
                    }
                }
            
            stats = {
                'total_memories': len(all_data['ids']),
                'tags': {},
                'types': {},
                'time_distribution': {
                    'last_24h': 0,
                    'last_week': 0,
                    'last_month': 0,
                    'older': 0
                }
            }
            
            current_time = time.time()
            
            for metadata in all_data['metadatas']:
                if metadata:
                    # Count tags
                    if 'tags_str' in metadata:
                        tags = metadata['tags_str'].split(',')
                        for tag in tags:
                            stats['tags'][tag] = stats['tags'].get(tag, 0) + 1
                    
                    # Count types
                    if 'type' in metadata:
                        mem_type = metadata['type']
                        stats['types'][mem_type] = stats['types'].get(mem_type, 0) + 1
                    
                    # Time distribution
                    if 'timestamp' in metadata:
                        age = current_time - float(metadata['timestamp'])
                        if age < 24 * 3600:
                            stats['time_distribution']['last_24h'] += 1
                        elif age < 7 * 24 * 3600:
                            stats['time_distribution']['last_week'] += 1
                        elif age < 30 * 24 * 3600:
                            stats['time_distribution']['last_month'] += 1
                        else:
                            stats['time_distribution']['older'] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats generation failed: {str(e)}")
            raise

    def optimize_database(self):
        """Optimize the database for better performance"""
        try:
            # Get all data from current collection
            all_data = self.collection.get()
            
            if not all_data or not all_data.get('ids'):
                logger.info("No data to optimize")
                return False
            
            # Create backup before optimization
            logger.info("Creating backup before optimization...")
            backup_path = self.create_backup()
            logger.info(f"Backup created at: {backup_path}")
            
            try:
                # Get all data again (in case it changed during backup)
                all_data = self.collection.get()
                
                # Delete and recreate the collection
                logger.info("Recreating collection...")
                collection_name = "memory_collection"
                
                try:
                    self.client.delete_collection(collection_name)
                except:
                    pass
                
                new_collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                # Add all data at once if small, otherwise batch
                if len(all_data['ids']) < 100:
                    new_collection.add(
                        ids=all_data['ids'],
                        documents=all_data['documents'],
                        embeddings=all_data['embeddings'],
                        metadatas=all_data['metadatas']
                    )
                else:
                    # Batch size for processing
                    batch_size = 100
                    
                    # Process in batches
                    for i in range(0, len(all_data['ids']), batch_size):
                        batch_ids = all_data['ids'][i:i + batch_size]
                        batch_documents = all_data['documents'][i:i + batch_size]
                        batch_embeddings = all_data['embeddings'][i:i + batch_size]
                        batch_metadatas = all_data['metadatas'][i:i + batch_size]
                        
                        new_collection.add(
                            ids=batch_ids,
                            documents=batch_documents,
                            embeddings=batch_embeddings,
                            metadatas=batch_metadatas
                        )
                
                # Update our reference to the new collection
                self.collection = new_collection
                
                logger.info("Database optimization completed successfully")
                return True
                
            except Exception as optimize_error:
                logger.error(f"Optimization failed, attempting to restore from backup: {str(optimize_error)}")
                self.restore_backup(backup_path)
                raise
            
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            raise