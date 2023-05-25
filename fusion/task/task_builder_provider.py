from .pretraining import PretrainingTaskBuilder
from .inference import InferenceTaskBuilder
from .linear_evaluation import LinearEvaluationTaskBuilder
from .logits_extraction import LogitsExtractionTaskBuilder
from .logreg_evaluation import LogRegEvaluationTaskBuilder
from .tsne import TsneTaskBuilder
from .misc import TaskId
from .saliency import SaliencyTaskBuilder
from .feature_extraction import FeatureExtractionTaskBuilder

from fusion.utils import ObjectProvider


task_builder_provider = ObjectProvider()
task_builder_provider.register_object(
    TaskId.PRETRAINING, PretrainingTaskBuilder
)
task_builder_provider.register_object(
    TaskId.LINEAR_EVALUATION, LinearEvaluationTaskBuilder
)
task_builder_provider.register_object(
    TaskId.LOGREG_EVALUATION, LogRegEvaluationTaskBuilder
)
task_builder_provider.register_object(
    TaskId.SALIENCY, SaliencyTaskBuilder
)
task_builder_provider.register_object(
    TaskId.TSNE, TsneTaskBuilder
)
task_builder_provider.register_object(
    TaskId.FEATURE_EXTRACTION, FeatureExtractionTaskBuilder
)
task_builder_provider.register_object(
    TaskId.INFERENCE, InferenceTaskBuilder
)
task_builder_provider.register_object(
    TaskId.LOGITS_EXTRACTION, LogitsExtractionTaskBuilder
)
