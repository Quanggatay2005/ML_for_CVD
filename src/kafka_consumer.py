import os
import json
import time
from datetime import datetime
import pandas as pd
from kafka import KafkaConsumer

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_ROOT, "data", "inference_logs")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print(f"Starting Kafka Consumer. Saving logs to {OUT_DIR}")
    try:
        consumer = KafkaConsumer(
            'inference_logs',
            bootstrap_servers=['localhost:9092'],
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            group_id='mlops-drift-monitor',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
    except Exception as e:
        print(f"Failed to connect to Kafka: {e}")
        return
    
    batch = []
    MAX_BATCH_SIZE = 50
    SAVE_INTERVAL = 15 # seconds
    LAST_SAVE_TIME = time.time()

    try:
        while True:
            # Poll for messages with 1s timeout
            msg_pack = consumer.poll(timeout_ms=1000)
            
            for tp, messages in msg_pack.items():
                for message in messages:
                    batch.append(message.value)
            
            # Flush batch if size reached or time elapsed
            if len(batch) >= MAX_BATCH_SIZE or (time.time() - LAST_SAVE_TIME) > SAVE_INTERVAL:
                if batch:
                    _save_batch(batch)
                    batch = []
                LAST_SAVE_TIME = time.time()
                
    except KeyboardInterrupt:
        print("\nStopping consumer...")
    finally:
        if batch:
            _save_batch(batch)
        try:
            consumer.close()
        except Exception:
            pass

def _save_batch(batch):
    # Flatten the features and prediction dictionaries
    flattened = []
    for item in batch:
        row = {"timestamp": item["timestamp"]}
        
        # Prefix feature columns with feat_
        for k, v in item.get("features", {}).items():
            row[f"feat_{k}"] = v
            
        # Prefix prediction columns with pred_
        for k, v in item.get("prediction", {}).items():
            row[f"pred_{k}"] = v
            
        flattened.append(row)
        
    df = pd.DataFrame(flattened)
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUT_DIR, f"log_{timestamp_str}_{len(df)}.parquet")
    
    df.to_parquet(filepath, engine='pyarrow', index=False)
    print(f"Saved {len(df)} records to {filepath}")

if __name__ == "__main__":
    main()
