"""AI Services for Deepiri"""
from .task_classifier import TaskClassifier, get_task_classifier
from .challenge_generator import ChallengeGenerator, get_challenge_generator
from .context_aware_adaptation import ContextAwareAdapter, get_context_adapter
from .multimodal_understanding import MultimodalTaskUnderstanding, get_multimodal_understanding
from .neuro_symbolic_challenge import NeuroSymbolicChallengeGenerator, get_neuro_symbolic_generator
from .advanced_task_parser import AdvancedTaskParser, get_advanced_task_parser
from .adaptive_challenge_generator import AdaptiveChallengeGenerator, get_adaptive_challenge_generator
from .hybrid_ai_service import HybridAIService, get_hybrid_ai_service
from .reward_model import RewardModelService, get_reward_model
from .embedding_service import EmbeddingService, get_embedding_service
from .inference_service import InferenceService, get_inference_service

# Cyrex Guard Services
from .invoice_parser import UniversalInvoiceProcessor, get_universal_invoice_processor, ProcessedInvoice, InvoiceLineItem
from .pricing_benchmark import PricingBenchmarkEngine, get_pricing_benchmark_engine, PricingBenchmark, PriceComparison, PricingTier
from .lora_loader import IndustryLoRAService, get_industry_lora_service, LoRAAdapterInfo, LoRAAdapterStatus
from .fraud_detector import UniversalFraudDetectionService, get_universal_fraud_detection_service, FraudDetectionResult, FraudIndicator

__all__ = [
    # Standard services
    'TaskClassifier',
    'get_task_classifier',
    'ChallengeGenerator',
    'get_challenge_generator',
    'ContextAwareAdapter',
    'get_context_adapter',
    'MultimodalTaskUnderstanding',
    'get_multimodal_understanding',
    'NeuroSymbolicChallengeGenerator',
    'get_neuro_symbolic_generator',
    # Advanced services
    'AdvancedTaskParser',
    'get_advanced_task_parser',
    'AdaptiveChallengeGenerator',
    'get_adaptive_challenge_generator',
    # Supporting services
    'HybridAIService',
    'get_hybrid_ai_service',
    'RewardModelService',
    'get_reward_model',
    'EmbeddingService',
    'get_embedding_service',
    'InferenceService',
    'get_inference_service',
    # Vendor Fraud Detection Services
    'UniversalInvoiceProcessor',
    'get_universal_invoice_processor',
    'ProcessedInvoice',
    'InvoiceLineItem',
    'PricingBenchmarkEngine',
    'get_pricing_benchmark_engine',
    'PricingBenchmark',
    'PriceComparison',
    'PricingTier',
    'IndustryLoRAService',
    'get_industry_lora_service',
    'LoRAAdapterInfo',
    'LoRAAdapterStatus',
    'UniversalFraudDetectionService',
    'get_universal_fraud_detection_service',
    'FraudDetectionResult',
    'FraudIndicator',
]

