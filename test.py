import ollama
import json

# 配置参数
MODEL_NAME = "deepseek-r1:32b"  # 确保你本地已运行 ollama pull deepseek-r1:7b

class Agent:
    def __init__(self, name, system_prompt):
        self.name = name
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": system_prompt}]

    def chat(self, user_input):
        """与该智能体进行对话"""
        self.history.append({"role": "user", "content": user_input})
        
        response = ollama.chat(
            model=MODEL_NAME,
            messages=self.history,
            options={'temperature': 0.8}
        )
        
        reply = response['message']['content']
        # 针对 deepseek-r1，有时会输出 <think> 标签内容，我们可以根据需要过滤或保留
        self.history.append({"role": "assistant", "content": reply})
        return reply

def run_multi_agent_session(topic, rounds=2):
    """运行多智能体互动会话"""
    
    # 定义两个分身：一个是充满创意的程序员，一个是严谨的架构师
    coder = Agent(
        name="Coder_Deep",
        system_prompt="你是一名精通 Python 的高级开发工程师。你的回答应该简洁、高效，并包含代码示例。"
    )
    
    reviewer = Agent(
        name="Reviewer_Deep",
        system_prompt="你是一名资深系统架构师和安全专家。你的任务是审查代码，指出潜在的 Bug、性能瓶颈或安全漏洞，并提出改进建议。"
    )

    print(f"=== 讨论主题: {topic} ===\n")
    
    # 初始输入
    current_message = f"请帮我实现一个关于 '{topic}' 的初步方案。"
    
    for i in range(rounds):
        print(f"--- 第 {i+1} 轮对话 ---")
        
        # 1. 开发者生成方案
        print(f"[{coder.name}] 正在思考方案...")
        coder_reply = coder.chat(current_message)
        print(f"\n{coder.name} 的回复:\n{coder_reply}\n")
        
        # 2. 审查员评价方案
        print(f"[{reviewer.name}] 正在审查方案...")
        reviewer_reply = reviewer.chat(f"这是开发者的方案，请进行审查并提出改进建议：\n{coder_reply}")
        print(f"\n{reviewer.name} 的审查意见:\n{reviewer_reply}\n")
        
        # 3. 将审查意见反馈给开发者进行下一轮迭代
        current_message = f"根据审查员的建议，请优化你的方案：\n{reviewer_reply}"
        
        print("-" * 50)

if __name__ == "__main__":
    # 启动交互
    target_topic = "高并发下的数据缓存策略"
    run_multi_agent_session(target_topic, rounds=2)