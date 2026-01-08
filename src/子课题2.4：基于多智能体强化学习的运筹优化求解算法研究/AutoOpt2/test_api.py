#!/usr/bin/env python3
"""
API配置测试脚本
用于验证config2.yaml中的API配置是否可用
"""

import os
import yaml
import asyncio
from openai import AsyncOpenAI
from pathlib import Path

def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "config" / "config2.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['llm']

async def test_api_with_proxy(llm_config):
    """使用代理测试API"""
    print("\n" + "="*60)
    print("测试1: 使用代理 (127.0.0.1:7890)")
    print("="*60)
    
    # 设置代理
    os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
    
    try:
        client = AsyncOpenAI(
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url']
        )
        
        print(f"📡 API类型: {llm_config['api_type']}")
        print(f"🤖 模型: {llm_config['model']}")
        print(f"🔗 Base URL: {llm_config['base_url']}")
        print(f"🔑 API Key: {llm_config['api_key'][:20]}...")
        print(f"🌐 代理: 127.0.0.1:7890")
        print("\n正在发送测试请求...")
        
        response = await client.chat.completions.create(
            model=llm_config['model'],
            messages=[
                {"role": "user", "content": "请用一句话介绍你自己"}
            ],
            timeout=30.0
        )
        
        print("\n✅ 使用代理连接成功！")
        print(f"📝 响应内容: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"\n❌ 使用代理连接失败")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        return False
    finally:
        # 清理代理环境变量
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)

async def test_api_without_proxy(llm_config):
    """不使用代理测试API"""
    print("\n" + "="*60)
    print("测试2: 不使用代理 (直连)")
    print("="*60)
    
    # 确保没有代理设置
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    
    try:
        client = AsyncOpenAI(
            api_key=llm_config['api_key'],
            base_url=llm_config['base_url']
        )
        
        print(f"📡 API类型: {llm_config['api_type']}")
        print(f"🤖 模型: {llm_config['model']}")
        print(f"🔗 Base URL: {llm_config['base_url']}")
        print(f"🔑 API Key: {llm_config['api_key'][:20]}...")
        print(f"🌐 代理: 无 (直连)")
        print("\n正在发送测试请求...")
        
        response = await client.chat.completions.create(
            model=llm_config['model'],
            messages=[
                {"role": "user", "content": "请用一句话介绍你自己"}
            ],
            timeout=30.0
        )
        
        print("\n✅ 直连成功！")
        print(f"📝 响应内容: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"\n❌ 直连失败")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        return False

async def main():
    print("\n" + "🔍 开始测试 API 配置".center(60, "="))
    
    try:
        # 加载配置
        llm_config = load_config()
        print("\n✓ 配置文件加载成功")
        
        # 测试1: 使用代理
        result_with_proxy = await test_api_with_proxy(llm_config)
        
        # 等待一下，避免请求过快
        await asyncio.sleep(1)
        
        # 测试2: 不使用代理
        result_without_proxy = await test_api_without_proxy(llm_config)
        
        # 总结
        print("\n" + "="*60)
        print("📊 测试结果总结")
        print("="*60)
        print(f"使用代理 (127.0.0.1:7890): {'✅ 可用' if result_with_proxy else '❌ 不可用'}")
        print(f"直连 (无代理):            {'✅ 可用' if result_without_proxy else '❌ 不可用'}")
        
        print("\n💡 建议:")
        if result_without_proxy and not result_with_proxy:
            print("   → 建议在 AutoOpt.py 中注释掉代理设置，使用直连")
            print("   → 修改 AutoOpt.py 第3-4行，在前面加 # 注释掉")
        elif result_with_proxy and not result_without_proxy:
            print("   → API需要通过代理访问，请确保代理服务正在运行")
        elif result_with_proxy and result_without_proxy:
            print("   → API在有代理和无代理情况下都可用")
            print("   → 如果是国内访问阿里云API，建议使用直连（不使用代理）")
        else:
            print("   → API配置可能有问题，请检查：")
            print("     1. API Key是否有效")
            print("     2. 模型名称是否正确")
            print("     3. Base URL是否正确")
            print("     4. 网络连接是否正常")
        
        print("\n" + "="*60 + "\n")
        
    except FileNotFoundError:
        print("\n❌ 错误: 找不到配置文件 config/config2.yaml")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

