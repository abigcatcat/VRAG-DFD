import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json
import torch

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""


# For additional reward functions, refer to swift/plugin/orm.py.
class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


orms['external_countdown'] = CountdownORM

class AccORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>([\s\S]*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                ground_truth = ground_truth.lower()
                if 'real' in ground_truth:
                    ground_truth_mapped, ground_truth_mapped_opp = ['real', 'authentic', 'genuine', 'camera', 'natural'], ['fake', 'synthetic', 'generated', 'machine', 'artificial', 'algorithmical']
                else:
                    ground_truth_mapped, ground_truth_mapped_opp = ['fake', 'synthetic', 'generated', 'machine', 'artificial', 'algorithmical'], ['real', 'authentic', 'genuine', 'camera', 'natural']
        
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>([\s\S]*?)</answer>', content, flags=re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = student_answer.lower()

                # Compare the extracted answers
                has_target = any(word in student_answer for word in ground_truth_mapped)
                has_target_oppo = any(word in student_answer for word in ground_truth_mapped_opp)
                if has_target and not has_target_oppo:
                    reward = 1.0

            except Exception:
                pass
            rewards.append(reward)
        return rewards
    
orms['bin_acc'] = AccORM

class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


orms['external_r1v_acc'] = MultiModalAccuracyORM


class MultiTurnThinkingTips(ORM):
    """
    A reward function example designed for use with the `ThinkingTipsScheduler`.

    This class demonstrates how to handle reward computation when a single
    training sample (or request) is split into multiple "turns" or steps.
    Specifically, it computes the reward based on the **last turn** of each
    multi-turn trajectory using a math accuracy function.

    NOTE
    ----
    If you feed fragments of the *same* trajectory as independent samples, this
    function **must return an identical reward for every fragment**
    """

    def __init__(self):
        from swift.plugin.orm import MathAccuracy
        self.acc_func = MathAccuracy()

    def __call__(self, completions, **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id')

        global_trajectorys: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs')

        rewards = []
        for local_tra_id in trajectory_ids:
            total_trajectory_inputs = global_trajectorys[local_tra_id]
            # For reward calculation, we use the entire trajectory of this sample.
            # Here, we specifically evaluate only the last turn.
            last_turn_messages = total_trajectory_inputs[-1]['messages']
            last_turn_completion = last_turn_messages[-1]['content']
            last_turn_solution = total_trajectory_inputs[-1]['solution']
            # Compute reward based on math accuracy for the final completion.
            reward = self.acc_func([last_turn_completion], [last_turn_solution])[0]
            rewards.append(reward)
        return rewards


orms['thinking_tips'] = MultiTurnThinkingTips

class Multi_Format(ORM):
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Reward function that checks if the completion strictly follows the V7 4-step format.
        1. <初步视觉分析>...</初步视觉分析>
        2. <RAG参考信息分析>...</RAG参考信息分析>
        3. <融合推理与决策>...</融合推理与决策>
        4. <answer>...</answer>
        """
        # 定义严格的四步格式正则
        # \s* 允许标签之间有空格或换行
        # [\s\S]*? 非贪婪匹配标签内的任意内容（包括换行符）
        pattern = (
            r"^\s*"  # 允许开头有空白
            r"<Preliminary Visual Analysis>[\s\S]*?</Preliminary Visual Analysis>\s*"
            r"<RAG Reference Information Analysis>[\s\S]*?</RAG Reference Information Analysis>\s*"
            r"<Fusion, Reasoning, and Decision>[\s\S]*?</Fusion, Reasoning, and Decision>\s*"
            r"<verdict>[\s\S]*?</verdict>"
            r"\s*$"  # 允许结尾有空白
        )
        rewards = []
        for content in completions:
            # 使用 re.IGNORECASE 增加一点容错性（虽然最好严格匹配大小写）
            match = re.match(pattern, content, flags=re.IGNORECASE)
            rewards.append(1.0 if match else 0.0)
        return rewards

orms['multi_format'] = Multi_Format

import re
from typing import List

# 假设 ORM 是一个基类
class ORM:
    pass 

class Balance_ORM(ORM):
    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        """
        Combined reward function:
        - Base reward (1.0) for ANY correct answer.
        - Bonus reward (+0.5) if the correct answer is 'REAL' (to fix bias).
        """
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            
            # 警告：在调试时，最好不要用 "except Exception: pass"
            # 否则你永远不知道哪里出错了。
            # 暂时注释掉，或者只 catch 特定错误
            # try:
            
            # =======================
            # 1. 解析标准答案 (GT)
            # =======================
            ground_truth = sol.strip().lower()
            sol_match = re.search(r'<verdict>(.*?)</verdict>', ground_truth, flags=re.DOTALL)
            if sol_match:
                ground_truth = sol_match.group(1).strip().lower()

            is_gt_real = 'real' in ground_truth
            is_gt_fake = 'fake' in ground_truth

            if is_gt_real:
                targets = ['real', 'authentic', 'genuine']
                opposites = ['fake', 'synthetic', 'generated', 'deepfake']
            elif is_gt_fake:
                targets = ['fake', 'synthetic', 'generated', 'deepfake']
                opposites = ['real', 'authentic', 'genuine']
            else:
                rewards.append(0.0)
                continue

            # =======================
            # 2. 解析模型预测 (Pred)
            # =======================
            
            # 使用 re.IGNORECASE 来忽略 <verdict> 标签的大小写
            content_match = re.findall(r'<verdict>(.*?)</verdict>', content, flags=re.IGNORECASE | re.DOTALL)

            # 3. (已修复) 只有在找到匹配项时才计算奖励
            if content_match:
                pred = content_match[-1].strip().lower()

                # (修复点) 检查奖励的逻辑 *必须* 缩进在 if content_match 内部
                has_target = any(t in pred for t in targets)
                has_opposite = any(o in pred for o in opposites)
                
                if has_target and not has_opposite:
                    # === 核心合并逻辑 ===
                    reward = 1.0  # 基础分：答对了！
                    
                    if is_gt_real:
                        reward += 0.5  # 额外分
            
            # (修复点) 如果 content_match 是空的 (即没找到 <verdict> 标签),
            # 代码会直接跳过上面的 if 块, reward 保持 0.0, 这是正确的。
            
            # except Exception:
            #     pass
            
            rewards.append(reward)
            
        return rewards

orms['balance_acc'] = Balance_ORM

class FinalMetaReward_v5(ORM):
    """
    V-Final v5: 您的（V-User-86）“非对称基础分” + “对称批判奖励”方案
    (已更新: “读心术”现在只扫描包含 "My preliminary visual analysis" 的那句话)
    """

    def __init__(self,         
                 # --- V-User-86 修正 ---
                 correct_base_reward_fake=1.0,
                 correct_base_reward_real=1.5, # Real 基础分高 0.5
                 
                 # --- V-User-48/53 逻辑 ---
                 critique_bonus_reward=1.0,  # 对称的批判奖金
                 wrong_penalty=-1.0,
                 critique_penalty=-1.0       # 对称的批判惩罚
                 ):
        # super().__init__(*args, **kwargs)
        
        self.correct_base_reward_fake = correct_base_reward_fake
        self.correct_base_reward_real = correct_base_reward_real
        self.critique_bonus_reward = critique_bonus_reward
        self.wrong_penalty = wrong_penalty
        self.critique_penalty = critique_penalty
        
        # V7 (V-Response-18) 中正确的“四标签” Rege
        
        # V-User-70: “读心术” Regex (保持 re.IGNORECASE)
        self.proxy_pattern_real = re.compile(
            # \b 确保我们匹配的是完整的单词
            r"\b(authentic|authenticity|natural|real|genuine|consistent)\b|(\bclear texture\b)",
            flags=re.IGNORECASE | re.DOTALL
        )

        self.proxy_pattern_fake = re.compile(
            # 修正：移除了开头的 | 和多余的 "unnatural"
            # \b 确保我们匹配的是完整的单词
            r"\b(artifact|artifacts|inconsistent|unnatural|synthetic|fake|unnaturally|blurry|mismatch)\b",
            flags=re.IGNORECASE | re.DOTALL
        )

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        for content, sol_json_str in zip(completions, solution):
            total_reward = 0.0
            
            try:
                sol_data = json.loads(sol_json_str)
                # --- 2. 解析 Solution (GT + RAG元数据) ---
                ground_truth_raw = sol_data.get('ground_truth')
                ground_truth = str(ground_truth_raw).lower() if ground_truth_raw is not None else ''
                print(f"Parsed ground_truth: {ground_truth!r}") # 使用 !r 打印更清晰
                rag_correct = sol_data.get('rag_correct', False) 
                is_gt_real = 'real' in ground_truth

                # --- 3. 解析 Prediction (Final Answer) ---
                content_match = re.findall(r'<verdict>(.*?)</verdict>', content, flags=re.IGNORECASE | re.DOTALL)
                if content_match:
                    pred_str = content_match[-1].strip().lower()
                else:
                    pred_str = "" # 默认为空字符串，不再崩溃
                print(f"content_match: {content_match}")
                
                is_prediction_correct = (pred_str == ground_truth)
                
                # --- 4. 动态“读心”（V-User-70）---
                s1_correct = False # 默认为错
                fusion_match = re.search(r"<Fusion, Reasoning, and Decision>([\s\S]*?)</Fusion, Reasoning, and Decision>", content, flags=re.IGNORECASE | re.DOTALL)
                
                if fusion_match:
                    # --- 【读心术逻辑重大修改】 ---
                    # 1. 获取 <Fusion> 块的全部文本
                    fusion_text = fusion_match.group(1).strip()
                    if fusion_text:
                        # 2. 定义触发短语
                        trigger_phrase = "my preliminary visual analysis"
                        
                        # 3. 按句子分割 (使用. ! ?)
                        sentences = re.split(r'[.!?\n]', fusion_text)
                        target_sentence = "" # 存储我们找到的目标句子

                        # 4. 遍历所有句子，找到第一个包含触发短语的句子
                        for sentence in sentences:
                            if trigger_phrase in sentence.lower():
                                target_sentence = sentence # 找到了，保留它
                                break # 立即停止，只取第一个
                        # 5. 如果找到了目标句子，*才*在其中搜索信号
                        if target_sentence:
                            # (我们搜索原始大小写的句子，因为 proxy_pattern 已经设置了 IGNORECASE)
                            s1_real_signal = self.proxy_pattern_real.search(target_sentence)

                            s1_fake_signal = self.proxy_pattern_fake.search(target_sentence)


                            if s1_real_signal and not s1_fake_signal:
                                s1_correct = (ground_truth == 'real')
                            elif s1_fake_signal and not s1_real_signal:
                                s1_correct = (ground_truth == 'fake')

                        # (如果没找到包含 trigger_phrase 的句子, s1_correct 保持 False)
                    # --- 【修改结束】 ---

                # --- 5. 执行“V-Final v5”奖励逻辑 ---
                is_conflict = (s1_correct != rag_correct)

                if is_prediction_correct:
                    # 答对了 (场景 1, 3, 5, 7)
                    if is_gt_real:
                        total_reward += self.correct_base_reward_real # +1.5
                    else:
                        total_reward += self.correct_base_reward_fake # +1.0
                    
                    if is_conflict:
                        # 场景 3 或 5：在冲突中答对了！
                        total_reward += self.critique_bonus_reward # +1.0
                
                else:
                    # 答错了 (场景 2, 4, 6, 8)
                    total_reward += self.wrong_penalty # -1.0
                    
                    if is_conflict:
                        # 场景 4 或 6：在冲突中答错了！
                        total_reward += self.critique_penalty # -1.0

            except Exception as e:
                # --- 【重要修复】 ---
                # 【调试增强】 捕获所有其他意外错误 (例如 JSON 损坏)
                # 按照您的要求，打印出导致失败的“这条数据”
                print("\n" + "="*20 + " REWARD FUNCTION ERROR " + "="*20)
                print(f"Warning: Reward function failed with UNEXPECTED error: {e}. Resetting reward to 0.0.")
                print(f"Failed solution string (这条数据): {sol_json_str}")
                print(f"Failed completion content (这条数据): {repr(content)}")
                print("="*60 + "\n")
                total_reward = 0.0
            
            rewards.append(total_reward)
        return rewards

class FinalMetaReward_v6(ORM):
    """
    V-Final v5 (已修正): 完全对称的基础分 + 对称批判奖惩方案
    逻辑：无论 Real 还是 Fake，基础奖惩分值完全一致。
    “读心术”现在只扫描包含 "My preliminary visual analysis" 的那句话。
    """

    def __init__(self,         
                 # --- 统一基础分 (去除了 1.5/1.0 的不平衡) ---
                 correct_base_reward=1.0,   # 答对的基础奖励
                 wrong_penalty=-1.0,         # 答错的基础惩罚
                 
                 # --- 对称的批判逻辑 ---
                 critique_bonus_reward=1.0,  # 在冲突中坚持正确的额外奖金
                 critique_penalty=-1.0       # 在冲突中被误导答错的额外惩罚
                 ):
        
        self.correct_base_reward = correct_base_reward
        self.wrong_penalty = wrong_penalty
        self.critique_bonus_reward = critique_bonus_reward
        self.critique_penalty = critique_penalty
        
        # V-User-70: “读心术” Regex (保持 re.IGNORECASE)
        self.proxy_pattern_real = re.compile(
            r"\b(authentic|authenticity|natural|real|genuine|consistent)\b|(\bclear texture\b)",
            flags=re.IGNORECASE | re.DOTALL
        )

        self.proxy_pattern_fake = re.compile(
            r"\b(artifact|artifacts|inconsistent|unnatural|synthetic|fake|unnaturally|blurry|mismatch)\b",
            flags=re.IGNORECASE | re.DOTALL
        )

    def __call__(self, completions: List[str], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        for content, sol_json_str in zip(completions, solution):
            total_reward = 0.0
            
            try:
                sol_data = json.loads(sol_json_str)
                ground_truth_raw = sol_data.get('ground_truth')
                ground_truth = str(ground_truth_raw).lower() if ground_truth_raw is not None else ''
                
                rag_correct = sol_data.get('rag_correct', False) 

                content_match = re.findall(r'<verdict>(.*?)</verdict>', content, flags=re.IGNORECASE | re.DOTALL)
                if content_match:
                    pred_str = content_match[-1].strip().lower()
                else:
                    pred_str = ""
                
                is_prediction_correct = (pred_str == ground_truth)
                
                s1_correct = False 
                fusion_match = re.search(r"<Fusion, Reasoning, and Decision>([\s\S]*?)</Fusion, Reasoning, and Decision>", content, flags=re.IGNORECASE | re.DOTALL)
                
                if fusion_match:
                    fusion_text = fusion_match.group(1).strip()
                    if fusion_text:
                        trigger_phrase = "my preliminary visual analysis"
                        # 按句子分割
                        sentences = re.split(r'[.!?\n]', fusion_text)
                        target_sentence = "" 

                        for sentence in sentences:
                            if trigger_phrase in sentence.lower():
                                target_sentence = sentence 
                                break 

                        if target_sentence:
                            s1_real_signal = self.proxy_pattern_real.search(target_sentence)
                            s1_fake_signal = self.proxy_pattern_fake.search(target_sentence)

                            # 判定初步视觉判断是否与事实一致
                            if s1_real_signal and not s1_fake_signal:
                                s1_correct = (ground_truth == 'real')
                            elif s1_fake_signal and not s1_real_signal:
                                s1_correct = (ground_truth == 'fake')

                is_conflict = (s1_correct != rag_correct)

                if is_prediction_correct:
                    total_reward += self.correct_base_reward 
                    
                    if is_conflict:
                        total_reward += self.critique_bonus_reward 
                
                else:
                    total_reward += self.wrong_penalty 
                    
                    if is_conflict:
                        total_reward += self.critique_penalty 

            except Exception as e:
                # 异常处理：打印导致错误的原始数据
                print("\n" + "="*20 + " REWARD FUNCTION ERROR " + "="*20)
                print(f"Warning: Unexpected error: {e}. Resetting reward to 0.0.")
                print(f"Failed solution string: {sol_json_str}")
                print(f"Failed completion content: {repr(content)}")
                print("="*60 + "\n")
                total_reward = 0.0
            
            rewards.append(total_reward)
            
        return rewards

orms['final_meta_reward'] = FinalMetaReward_v6




# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


orms['external_code_reward'] = CodeReward


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


orms['external_code_format'] = CodeFormat


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


orms['external_code_reward_by_judge0'] = CodeRewardByJudge0


# ref implementation: https://github.com/qiancheng0/ToolRL/blob/main/verl/utils/reward_score/rlla.py
# arxiv paper: https://arxiv.org/abs/2504.13958
# MAX1STEP30MAX3: enable Two stage reward Setting include Format and Correctness
# SCHEDULEREWARD: enable Dynamic (Finegrained) reward Setting include Format and Correctness
# Correctness Reward Granularity:
# COARSEREWARD -> Coarse, INTERMEDIATEREWARD -> Intermediate, REFINEDREWARD -> Finegrained
class ToolUseFormatReward(ORM):

    def __init__(self):
        self.format_max_possible = 1.0
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible
        # Two stage (Coarse) Setting, divide training into two phases. Format Reward in [0,0.5] if step < 30 else [0,1]
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2
            else:
                max_possible_reward = self.format_max_possible
                min_possible_reward = self.format_min_possible

        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        rewards = []
        responses = completions

        for response, ans in zip(responses, solution):
            reward = min_possible_reward
            if '<response>' in ans and '<tool_call>' not in ans:
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<response>') == 1 and response.count('</response>') == 1:
                    reward = max_possible_reward
            elif '<response>' not in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<tool_call>') == 1 and response.count('</tool_call>') == 1:
                    reward = max_possible_reward
            elif '<response>' in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if (re.search(pattern, response, re.DOTALL) and response.count('<tool_call>') == 1
                        and response.count('</tool_call>') == 1 and response.count('<response>') == 1
                        and response.count('</response>') == 1):
                    reward = max_possible_reward
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_format_reward'] = ToolUseFormatReward


class ToolUseLengthReward(ORM):

    def __init__(self):
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            max_reward_len = 512
        """Reward function that gives higher scores to longer completions."""
        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            if reward > 1.0:
                reward = 1.0

            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            rewards.append(final_reward)

        return rewards


orms['external_tooluse_length_reward'] = ToolUseLengthReward


class ToolUseCorrectnessReward(ORM):

    def __init__(self):
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    def match_score(self, list1, list2):
        if list1 == list2:
            return 1.0

        if os.getenv('REFINEDREWARD', 0) == '1':
            if list1 != list2:
                return 0.0

        if not list1 or not list2:
            return 0.0

        count1 = Counter(list1)  # Frequency count for list1
        count2 = Counter(list2)  # Frequency count for list2

        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection

        return intersection / max_possible if max_possible > 0 else 0.0

    def compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        if gt_tools == pd_tools:
            return max_possible_reward

        if os.getenv('COARSEREWARD', 0) == '1':
            if gt_tools != pd_tools:
                return min_possible_reward

        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self.match_score(list(gt_names), list(pd_names))

        local_max_possible = 1.0
        used_pd_indices = set()  # Keep track of matched pd_tools

        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']

            if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)

            best_match = None
            best_match_score = 0.0
            best_match_index = -1

            # Find the best matching unused pd_tool
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue

                if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                    if gt_tool == pd_tool:
                        best_match = pd_tool
                        best_match_index = i
                        best_match_score = 1.0
                        break
                    else:
                        continue

                pd_params = pd_tool['parameters']
                param_score = self.match_score(list(gt_params.keys()), list(pd_params.keys()))

                # Calculate correctness score for parameter values
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match = pd_tool
                    best_match_index = i

            if best_match:
                used_pd_indices.add(best_match_index)
                score += best_match_score

        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    # custoimzed reward functions: tool call correctness
    def __call__(self, completions, solution, **kwargs):
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible
        # two stage (Coarse) Setting, divide training into two phases.
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step < 30:
                max_possible_reward = max_possible_reward / 3
                min_possible_reward = min_possible_reward / 3
            else:
                max_possible_reward = max_possible_reward
                min_possible_reward = min_possible_reward
        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = (max_possible_reward - 2) * global_step / 150 + 2
            min_possible_reward = (min_possible_reward + 2) * global_step / 150 - 2
            if max_possible_reward > 3.0:
                max_possible_reward = 3.0
            if min_possible_reward < -3.0:
                min_possible_reward = -3.0

        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            reward = 0.0

            if '<tool_call>' not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = gt_tool_call.split('\n')
            gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

            try:
                # if the format is not correct, directly give the lowest possible score
                assert '<tool_call>' in response
                assert '</tool_call>' in response
                pd_tools = response.split('<tool_call>')[1].split('</tool_call>')[0].strip().split('\n')
                pd_tools = [json.loads(tool) for tool in pd_tools]
                reward = self.compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward,
                                                       min_possible_reward)  # top reward is 2
            except (ValueError, IndexError, AssertionError):
                reward = min_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward
"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs, **kwargs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step (Required): Constructs the next round of the infer request.
            - check_finished (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
            or override run method in MultiTurnScheduler class.

        Both methods accept:
            - the last turn's InferRequest/response_choice
            - the current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        swift rollout \
            --external_plugins /path/to/plugin.py \
            --multi_turn_scheduler my_scheduler
"""


class ToolCallScheduler(MultiTurnScheduler):
    # A simple scheduler that supports tool calls by overriding the `step` method
    # Tool parsing uses the ReAct format
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A simple tool registry. Extend or replace with your own tools as needed.
        self.tools = {
            'calculator': self._calculator_tool,
        }

    def _calculator_tool(self, expression: str) -> str:
        # A very small sandboxed calculator
        # The calculator tool implemented here can perform only basic arithmetic operations and
        # may not be able to solve all math problems in the dataset.
        import ast
        import operator

        def _evaluate_ast_node(node) -> Union[int, float]:
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                else:
                    raise TypeError(f'Unsupported constant type: {type(node.value)}')

            elif isinstance(node, ast.Num):
                return node.n

            elif isinstance(node, ast.BinOp):
                left = _evaluate_ast_node(node.left)
                right = _evaluate_ast_node(node.right)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported operation: {type(node.op).__name__}')

                if isinstance(node.op, ast.Div) and right == 0:
                    raise ZeroDivisionError('Division by zero')

                return op(left, right)

            elif isinstance(node, ast.UnaryOp):
                operand = _evaluate_ast_node(node.operand)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported unary operation: {type(node.op).__name__}')

                return op(operand)

            else:
                raise TypeError(f'Unsupported AST node type: {type(node).__name__}')

        try:
            expression = expression.strip().replace(' ', '')

            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return 'Error: expression contains disallowed characters.'

            if expression.count('(') != expression.count(')'):
                return 'Error: unmatched parentheses.'

            try:
                result = ast.literal_eval(expression)
                return f'Result: {result}'
            except (ValueError, SyntaxError):
                node = ast.parse(expression, mode='eval')
                result = _evaluate_ast_node(node.body)
                return f'Result: {result}'

        except Exception as e:
            return f'Calculation error: {e}'

    def _extract_tool_calls(self, text: str):
        """
        Parse tool-call patterns using ReAct format from model output.
        Format: Action: tool_name\nAction Input: parameters
        """
        import re

        pattern = r'Action:\s*(.*?)\s*\nAction Input:\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return [{'tool': name.strip(), 'params': params.strip()} for name, params in matches]

    def _execute_tools(self, tool_calls):
        """Run each requested tool and collect its observation string."""
        results = []
        for call in tool_calls:
            name, params = call['tool'], call['params']
            if name in self.tools:
                try:
                    result = self.tools[name](params)
                    results.append(result)
                except Exception as e:
                    results.append(f'tool error {e}')
            else:
                results.append(f'unknown tool {name}')
        return results

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        completion = response_choice.message.content
        tool_calls = self._extract_tool_calls(completion)
        if tool_calls is None:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)
        tool_calls = self._extract_tool_calls(completion)
        # assert len(tool_calls) == 1, 'this scheduler is designed for one tool call per turn'
        tool_results = self._execute_tools(tool_calls)
        # append tool result to the completion
        infer_request.messages[-1]['content'] += (tool_results[0])

        tokenizer = self.infer_engine.default_template.tokenizer
        result_tokens = tokenizer.encode(tool_results[0], add_special_tokens=False)
        token_ids.extend(result_tokens)
        loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            'rollout_infos': {
                'tool_results': tool_results[0],
                'num_turns': current_turn,
            }
        }


multi_turns['tool_call_scheduler'] = ToolCallScheduler


# register GYM env
class CustomEnv(Env):
    pass


envs['custom_env'] = CustomEnv


class CustomCtxManager(ContextManager):
    pass


context_managers['custom_ctx'] = CustomCtxManager
