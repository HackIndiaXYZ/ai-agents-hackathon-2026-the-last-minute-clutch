import asyncio
import sys
from nyayaeval.config.settings import get_settings
from nyayaeval.connectors.adaption_client import AdaptiveDataClient
from nyayaeval.connectors.neo4j_client import NyayaNeo4jClient
from nyayaeval.connectors.redis_client import NyayaRedisClient
from nyayaeval.connectors.registry import register_adaption, register_neo4j, register_redis
from nyayaeval.pipeline.checkpointer import get_checkpointer
from nyayaeval.pipeline.graph import compile_pipeline
from langchain_core.messages import HumanMessage

async def main():
    print("--- Initializing Infrastructure ---")
    settings = get_settings()
    
    neo4j = NyayaNeo4jClient(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
    await neo4j.connect()
    register_neo4j(neo4j)
    print("[OK] Neo4j connected")
    
    redis = NyayaRedisClient(url=settings.redis_url)
    await redis.connect()
    register_redis(redis)
    print("[OK] Redis connected")
    
    adaption = AdaptiveDataClient(api_key=settings.adaption_api_key, timeout=settings.adaption_api_timeout)
    await adaption.connect()
    register_adaption(adaption)
    print("[OK] Adaption SDK connected")
    
    pipeline = compile_pipeline(checkpointer=get_checkpointer())
    print("[OK] Pipeline compiled")
    
    print("\n--- Running Pipeline ---")
    
    test_document = "अभियुक्त ने धारा ३०२ के तहत हत्या की है। उसे आजीवन कारावास की सजा सुनाई जाती है।"
    
    # Run the pipeline
    config = {"configurable": {"thread_id": "test_thread_1"}}
    inputs = {
        "raw_text": test_document,
        "source_language": "hi",
        "document_id": "test_doc_001"
    }
    
    try:
        async for output in pipeline.astream(inputs, config, stream_mode="updates"):
            for node_name, state_update in output.items():
                print(f"[{node_name}] -> {state_update.get('current_phase', 'unknown')}")
                if "execution_logs" in state_update and state_update["execution_logs"]:
                    for log in state_update["execution_logs"]:
                        print(f"  | {log}")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        print("\n--- Cleaning up ---")
        await neo4j.close()
        await redis.close()
        await adaption.close()

if __name__ == "__main__":
    asyncio.run(main())
