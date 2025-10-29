"""
Simple Tool Call Accuracy Metric
Inspired by RAGAS ToolCallAccuracy but simplified for practical use
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import warnings


@dataclass
class ToolCall:
    """Represents a single tool call"""
    name: str
    args: Dict[str, Any]


@dataclass
class ToolCallAccuracyResult:
    """Evaluation result"""
    score: float  # 0.0 to 1.0
    sequence_aligned: bool
    arg_accuracy: float
    matched_tools: int
    total_tools: int


class ToolCallAccuracy:
    """
    Evaluates tool calling accuracy with argument comparison
    
    Usage:
        evaluator = ToolCallAccuracy(strict_order=True)
        
        predicted = [
            ToolCall(name="search", args={"query": "weather"}),
            ToolCall(name="calculator", args={"expr": "2+2"})
        ]
        
        reference = [
            ToolCall(name="search", args={"query": "weather"}),
            ToolCall(name="calculator", args={"expr": "2+2"})
        ]
        
        result = evaluator.evaluate(predicted, reference)
        print(f"Score: {result.score}")
    """
    
    def __init__(self, strict_order: bool = True):
        """
        Args:
            strict_order: If True, tool calls must be in exact order.
                         If False, order doesn't matter (parallel execution).
        """
        self.strict_order = strict_order
    
    def evaluate(
        self,
        predicted: List[ToolCall],
        reference: List[ToolCall]
    ) -> ToolCallAccuracyResult:
        """
        Evaluate predicted tool calls against reference
        
        Args:
            predicted: Tool calls made by the agent
            reference: Expected tool calls (ground truth)
        
        Returns:
            ToolCallAccuracyResult with score and details
        """
        
        # Handle empty cases
        if not predicted and not reference:
            return ToolCallAccuracyResult(1.0, True, 1.0, 0, 0)
        
        if not predicted:
            return ToolCallAccuracyResult(0.0, False, 0.0, 0, len(reference))
        
        if not reference:
            return ToolCallAccuracyResult(0.0, False, 0.0, 0, 0)
        
        # Sort if flexible ordering
        if not self.strict_order:
            predicted = sorted(predicted, key=lambda tc: (tc.name, str(tc.args)))
            reference = sorted(reference, key=lambda tc: (tc.name, str(tc.args)))
        
        # Warn on length mismatch
        if len(predicted) != len(reference):
            warnings.warn(
                f"Length mismatch: {len(predicted)} predicted vs {len(reference)} reference"
            )
        
        # Check sequence alignment
        pred_names = [tc.name for tc in predicted]
        ref_names = [tc.name for tc in reference]
        
        if self.strict_order:
            sequence_aligned = pred_names == ref_names
        else:
            sequence_aligned = sorted(pred_names) == sorted(ref_names)
        
        # Calculate argument accuracy
        matched = 0
        arg_scores = []
        
        compare_count = min(len(predicted), len(reference))
        
        for pred, ref in zip(predicted[:compare_count], reference[:compare_count]):
            if pred.name == ref.name:
                arg_score = self._compare_args(pred.args, ref.args)
                arg_scores.append(arg_score)
                if arg_score == 1.0:
                    matched += 1
            else:
                arg_scores.append(0.0)
        
        # Average argument accuracy
        avg_arg_accuracy = sum(arg_scores) / len(reference) if reference else 0.0
        
        # Coverage penalty for length mismatch
        coverage = compare_count / len(reference)
        
        # Final score
        score = avg_arg_accuracy * int(sequence_aligned) * coverage
        
        return ToolCallAccuracyResult(
            score=score,
            sequence_aligned=sequence_aligned,
            arg_accuracy=avg_arg_accuracy,
            matched_tools=matched,
            total_tools=len(reference)
        )
    
    def _compare_args(self, pred_args: Dict[str, Any], ref_args: Dict[str, Any]) -> float:
        """Compare arguments with exact string matching"""
        if not ref_args and not pred_args:
            return 1.0
        if not ref_args:
            return 0.0
        
        matches = 0
        for key in ref_args.keys():
            if key in pred_args and str(pred_args[key]) == str(ref_args[key]):
                matches += 1
        
        return matches / len(ref_args)


# ==================== Convenience Functions ====================

def evaluate_tool_calls(
    predicted: List[Dict[str, Any]],
    reference: List[Dict[str, Any]],
    strict_order: bool = True
) -> float:
    """
    Quick evaluation function
    
    Args:
        predicted: List of dicts with 'name' and 'args' keys
        reference: List of dicts with 'name' and 'args' keys
        strict_order: Whether order matters
    
    Returns:
        Score from 0.0 to 1.0
    
    Example:
        score = evaluate_tool_calls(
            predicted=[
                {"name": "search", "args": {"query": "weather"}},
                {"name": "calculator", "args": {"expr": "2+2"}}
            ],
            reference=[
                {"name": "search", "args": {"query": "weather"}},
                {"name": "calculator", "args": {"expr": "2+2"}}
            ]
        )
    """
    pred_calls = [ToolCall(name=p['name'], args=p.get('args', {})) for p in predicted]
    ref_calls = [ToolCall(name=r['name'], args=r.get('args', {})) for r in reference]
    
    evaluator = ToolCallAccuracy(strict_order=strict_order)
    result = evaluator.evaluate(pred_calls, ref_calls)
    
    return result.score


# ==================== Examples ====================

def example_perfect_match():
    """Example: Perfect match"""
    evaluator = ToolCallAccuracy(strict_order=True)
    
    predicted = [
        ToolCall(name="search", args={"query": "weather Paris"}),
        ToolCall(name="calculator", args={"expr": "15 * 2"})
    ]
    
    reference = [
        ToolCall(name="search", args={"query": "weather Paris"}),
        ToolCall(name="calculator", args={"expr": "15 * 2"})
    ]
    
    result = evaluator.evaluate(predicted, reference)
    
    print("Example 1: Perfect Match")
    print(f"  Score: {result.score:.2f}")
    print(f"  Sequence Aligned: {result.sequence_aligned}")
    print(f"  Arg Accuracy: {result.arg_accuracy:.2f}")
    print()


def example_wrong_args():
    """Example: Correct tools but wrong arguments"""
    evaluator = ToolCallAccuracy(strict_order=True)
    
    predicted = [
        ToolCall(name="search", args={"query": "weather NYC"}),  # Wrong city!
    ]
    
    reference = [
        ToolCall(name="search", args={"query": "weather Paris"}),
    ]
    
    result = evaluator.evaluate(predicted, reference)
    
    print("Example 2: Wrong Arguments")
    print(f"  Score: {result.score:.2f}")
    print(f"  Arg Accuracy: {result.arg_accuracy:.2f}")
    print()


def example_flexible_order():
    """Example: Tools in different order"""
    evaluator = ToolCallAccuracy(strict_order=False)
    
    predicted = [
        ToolCall(name="calculator", args={"expr": "2+2"}),
        ToolCall(name="search", args={"query": "weather"}),
    ]
    
    reference = [
        ToolCall(name="search", args={"query": "weather"}),
        ToolCall(name="calculator", args={"expr": "2+2"}),
    ]
    
    result = evaluator.evaluate(predicted, reference)
    
    print("Example 3: Flexible Order (strict_order=False)")
    print(f"  Score: {result.score:.2f}")
    print(f"  Sequence Aligned: {result.sequence_aligned}")
    print()


def example_missing_tools():
    """Example: Missing some tools"""
    evaluator = ToolCallAccuracy(strict_order=True)
    
    predicted = [
        ToolCall(name="search", args={"query": "weather"}),
    ]
    
    reference = [
        ToolCall(name="search", args={"query": "weather"}),
        ToolCall(name="calculator", args={"expr": "2+2"}),
        ToolCall(name="calendar", args={"date": "today"}),
    ]
    
    result = evaluator.evaluate(predicted, reference)
    
    print("Example 4: Missing Tools")
    print(f"  Score: {result.score:.2f}")
    print(f"  Matched: {result.matched_tools}/{result.total_tools}")
    print()


def example_convenience_function():
    """Example: Using convenience function"""
    score = evaluate_tool_calls(
        predicted=[
            {"name": "search", "args": {"query": "weather"}},
            {"name": "calculator", "args": {"expr": "2+2"}}
        ],
        reference=[
            {"name": "search", "args": {"query": "weather"}},
            {"name": "calculator", "args": {"expr": "2+2"}}
        ]
    )
    
    print("Example 5: Convenience Function")
    print(f"  Score: {score:.2f}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Tool Call Accuracy - Examples")
    print("=" * 60)
    print()
    
    example_perfect_match()
    example_wrong_args()
    example_flexible_order()
    example_missing_tools()
    example_convenience_function()