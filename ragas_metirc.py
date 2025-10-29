"""
Simple Single-Turn Tool Call Accuracy Metric for RAGAS
Drop-in replacement that works with ragas.evaluate()
"""

from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

try:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics.base import MetricType, SingleTurnMetric
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    SingleTurnSample = None
    MetricType = None
    SingleTurnMetric = object

if t.TYPE_CHECKING:
    from ragas.callbacks import Callbacks


@dataclass
class ToolCall:
    """Simple tool call representation"""
    name: str
    args: dict


class ToolCallAccuracyMetric(SingleTurnMetric if RAGAS_AVAILABLE else object):
    """
    Single-turn tool call accuracy metric for RAGAS
    
    Usage:
        from ragas import evaluate
        from ragas_metric import ToolCallAccuracyMetric
        
        metric = ToolCallAccuracyMetric(strict_order=True)
        
        results = evaluate(
            dataset=your_dataset,
            metrics=[metric]
        )
        
        print(results['tool_call_accuracy'])
    
    Dataset Requirements:
        - predicted_tool_calls: List of dicts with 'name' and 'args'
        - reference_tool_calls: List of dicts with 'name' and 'args'
    
    Example data:
        {
            "predicted_tool_calls": [
                {"name": "search", "args": {"query": "weather"}},
                {"name": "calculator", "args": {"expr": "2+2"}}
            ],
            "reference_tool_calls": [
                {"name": "search", "args": {"query": "weather"}},
                {"name": "calculator", "args": {"expr": "2+2"}}
            ]
        }
    """
    
    name: str = "tool_call_accuracy"
    
    def __init__(self, strict_order: bool = True):
        """
        Args:
            strict_order: If True, tool calls must be in exact order.
                         If False, order doesn't matter.
        """
        self.strict_order = strict_order
        
        if RAGAS_AVAILABLE:
            self._required_columns = {
                MetricType.SINGLE_TURN: {
                    "predicted_tool_calls",
                    "reference_tool_calls"
                }
            }
    
    def init(self, run_config):
        """Initialize (required by RAGAS)"""
        pass
    
    def _convert_to_tool_call(self, tool_dict: dict) -> ToolCall:
        """Convert dict to ToolCall"""
        return ToolCall(
            name=tool_dict.get('name', ''),
            args=tool_dict.get('args', {})
        )
    
    def _compare_args(self, pred_args: dict, ref_args: dict) -> float:
        """Compare arguments - exact string matching"""
        if not ref_args and not pred_args:
            return 1.0
        if not ref_args:
            return 0.0
        
        matches = 0
        for key in ref_args.keys():
            if key in pred_args and str(pred_args[key]) == str(ref_args[key]):
                matches += 1
        
        return matches / len(ref_args)
    
    def _evaluate(
        self,
        predicted: list[ToolCall],
        reference: list[ToolCall]
    ) -> float:
        """Core evaluation logic"""
        
        # Handle empty cases
        if not predicted and not reference:
            return 1.0
        if not predicted or not reference:
            return 0.0
        
        # Sort if flexible ordering
        if not self.strict_order:
            predicted = sorted(predicted, key=lambda tc: (tc.name, str(tc.args)))
            reference = sorted(reference, key=lambda tc: (tc.name, str(tc.args)))
        
        # Check sequence alignment
        pred_names = [tc.name for tc in predicted]
        ref_names = [tc.name for tc in reference]
        
        if self.strict_order:
            sequence_aligned = pred_names == ref_names
        else:
            sequence_aligned = sorted(pred_names) == sorted(ref_names)
        
        if not sequence_aligned:
            return 0.0
        
        # Calculate argument accuracy
        arg_scores = []
        compare_count = min(len(predicted), len(reference))
        
        for pred, ref in zip(predicted[:compare_count], reference[:compare_count]):
            if pred.name == ref.name:
                arg_score = self._compare_args(pred.args, ref.args)
                arg_scores.append(arg_score)
            else:
                arg_scores.append(0.0)
        
        # Average with coverage penalty
        avg_score = sum(arg_scores) / len(reference) if reference else 0.0
        coverage = compare_count / len(reference)
        
        return avg_score * coverage
    
    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks
    ) -> float:
        """Score a single turn sample (required by RAGAS)"""
        
        # Extract tool calls from sample
        predicted_dicts = sample.predicted_tool_calls or []
        reference_dicts = sample.reference_tool_calls or []
        
        # Convert to ToolCall objects
        predicted = [self._convert_to_tool_call(tc) for tc in predicted_dicts]
        reference = [self._convert_to_tool_call(tc) for tc in reference_dicts]
        
        # Evaluate
        return self._evaluate(predicted, reference)
    
    async def _ascore(self, row: dict, callbacks: Callbacks) -> float:
        """Score a row (required by RAGAS)"""
        return await self._single_turn_ascore(SingleTurnSample(**row), callbacks)


# ==================== Example Usage ====================

def example_basic_usage():
    """Basic usage example"""
    
    print("=" * 70)
    print("Basic Usage Example")
    print("=" * 70)
    print()
    
    if not RAGAS_AVAILABLE:
        print("⚠️  RAGAS not installed. Install with: pip install ragas")
        print("This example shows the data format you'll need.")
        print()
    
    # Sample data
    data = {
        "predicted_tool_calls": [
            {"name": "search", "args": {"query": "weather Paris"}},
            {"name": "calculator", "args": {"expr": "15 * 2"}}
        ],
        "reference_tool_calls": [
            {"name": "search", "args": {"query": "weather Paris"}},
            {"name": "calculator", "args": {"expr": "15 * 2"}}
        ]
    }
    
    print("Sample Data:")
    print(f"Predicted: {data['predicted_tool_calls']}")
    print(f"Reference: {data['reference_tool_calls']}")
    print()
    
    if RAGAS_AVAILABLE:
        # Run evaluation
        metric = ToolCallAccuracyMetric(strict_order=True)
        sample = SingleTurnSample(**data)
        
        import asyncio
        score = asyncio.run(metric._single_turn_ascore(sample, callbacks=None))
        
        print(f"Score: {score:.2f}")
        print()


def example_ragas_evaluate():
    """Full RAGAS evaluate example"""
    
    print("=" * 70)
    print("RAGAS Evaluate Integration")
    print("=" * 70)
    print()
    
    if not RAGAS_AVAILABLE:
        print("⚠️  RAGAS not installed.")
        print("Install with: pip install ragas")
        print()
        print("Once installed, you can use it like this:")
        print()
    
    print("""
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from datasets import Dataset
from ragas_metric import ToolCallAccuracyMetric

# Prepare your data
data = {
    "predicted_tool_calls": [
        [
            {"name": "search", "args": {"query": "weather"}},
            {"name": "calculator", "args": {"expr": "2+2"}}
        ],
        [
            {"name": "search", "args": {"query": "NYC weather"}},
        ]
    ],
    "reference_tool_calls": [
        [
            {"name": "search", "args": {"query": "weather"}},
            {"name": "calculator", "args": {"expr": "2+2"}}
        ],
        [
            {"name": "search", "args": {"query": "Paris weather"}},  # Wrong!
        ]
    ]
}

# Create dataset
dataset = Dataset.from_dict(data)
eval_dataset = EvaluationDataset.from_hf_dataset(dataset)

# Initialize metric
metric = ToolCallAccuracyMetric(strict_order=True)

# Evaluate
results = evaluate(
    dataset=eval_dataset,
    metrics=[metric]
)

# Results
print(results)
# Output: {'tool_call_accuracy': 0.5}
# (First sample: 1.0, Second sample: 0.0, Average: 0.5)
""")
    
    # Try to actually run it if RAGAS is available
    if RAGAS_AVAILABLE:
        try:
            from ragas import evaluate
            from ragas.dataset_schema import EvaluationDataset
            from datasets import Dataset
            
            print("\n✓ RAGAS is installed! Running actual evaluation...")
            print()
            
            data = {
                "predicted_tool_calls": [
                    [
                        {"name": "search", "args": {"query": "weather"}},
                        {"name": "calculator", "args": {"expr": "2+2"}}
                    ],
                    [
                        {"name": "search", "args": {"query": "NYC weather"}},
                    ]
                ],
                "reference_tool_calls": [
                    [
                        {"name": "search", "args": {"query": "weather"}},
                        {"name": "calculator", "args": {"expr": "2+2"}}
                    ],
                    [
                        {"name": "search", "args": {"query": "Paris weather"}},
                    ]
                ]
            }
            
            dataset = Dataset.from_dict(data)
            eval_dataset = EvaluationDataset.from_hf_dataset(dataset)
            
            metric = ToolCallAccuracyMetric(strict_order=True)
            
            results = evaluate(
                dataset=eval_dataset,
                metrics=[metric]
            )
            
            print("Results:")
            print(f"  Tool Call Accuracy: {results['tool_call_accuracy']:.2f}")
            print()
            
        except Exception as e:
            print(f"Error running evaluation: {e}")
            print()


def print_quick_start():
    """Print quick start guide"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                         Quick Start Guide                             ║
╚══════════════════════════════════════════════════════════════════════╝

1. Install RAGAS
   ──────────────
   pip install ragas

2. Import the Metric
   ──────────────────
   from ragas_metric import ToolCallAccuracyMetric

3. Prepare Your Data
   ──────────────────
   data = {
       "predicted_tool_calls": [
           [{"name": "tool1", "args": {"x": 1}}]
       ],
       "reference_tool_calls": [
           [{"name": "tool1", "args": {"x": 1}}]
       ]
   }

4. Evaluate
   ─────────
   from ragas import evaluate
   from datasets import Dataset
   from ragas.dataset_schema import EvaluationDataset
   
   dataset = Dataset.from_dict(data)
   eval_dataset = EvaluationDataset.from_hf_dataset(dataset)
   
   metric = ToolCallAccuracyMetric(strict_order=True)
   
   results = evaluate(
       dataset=eval_dataset,
       metrics=[metric]
   )
   
   print(results['tool_call_accuracy'])

╔══════════════════════════════════════════════════════════════════════╗
║                     Data Format (Important!)                          ║
╚══════════════════════════════════════════════════════════════════════╝

Each row in your dataset needs:

✓ predicted_tool_calls: list of dicts
  [
      {"name": "tool_name", "args": {"param1": "value1"}},
      {"name": "tool_name2", "args": {"param2": "value2"}}
  ]

✓ reference_tool_calls: list of dicts (same format)
  [
      {"name": "tool_name", "args": {"param1": "value1"}},
      {"name": "tool_name2", "args": {"param2": "value2"}}
  ]

╔══════════════════════════════════════════════════════════════════════╗
║                    Configuration Options                              ║
╚══════════════════════════════════════════════════════════════════════╝

strict_order=True  (default)
  → Tools must be called in exact order
  → Use for sequential workflows

strict_order=False
  → Tools can be in any order
  → Use for parallel execution

╔══════════════════════════════════════════════════════════════════════╗
║                    Combine with Other Metrics                         ║
╚══════════════════════════════════════════════════════════════════════╝

from ragas.metrics import Faithfulness, AnswerRelevancy

results = evaluate(
    dataset=eval_dataset,
    metrics=[
        ToolCallAccuracyMetric(strict_order=True),  # Your metric
        Faithfulness(),                              # RAGAS built-in
        AnswerRelevancy()                            # RAGAS built-in
    ]
)

""")


if __name__ == "__main__":
    print_quick_start()
    example_basic_usage()
    example_ragas_evaluate()