"""
Simple test to debug configuration issues
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.engine_graph_lora.arguments import EngineGraphLoRAArguments

def test_basic_config():
    """Test basic configuration creation"""
    
    print("Testing basic configuration creation...")
    
    # Create minimal test arguments
    test_args = [
        '--dataset', 'synthetic',
        '--llm_model_name', 'prajjwal1/bert-tiny',
        '--device', 'cpu'
    ]
    
    try:
        args_parser = EngineGraphLoRAArguments()
        config = args_parser.parse_args(test_args)
        
        print("✅ Config created successfully!")
        print(f"Config type: {type(config)}")
        print(f"Config attributes: {dir(config)}")
        
        if hasattr(config, 'device'):
            print(f"Device: {config.device}")
        else:
            print("❌ No device attribute found")
            
        if hasattr(config, 'llm'):
            print(f"LLM config type: {type(config.llm)}")
            if hasattr(config.llm, 'model_name'):
                print(f"LLM model: {config.llm.model_name}")
        else:
            print("❌ No llm attribute found")
            
        return True
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_config()
    print("Success!" if success else "Failed!") 