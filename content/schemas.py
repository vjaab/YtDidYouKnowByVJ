from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


class DifficultyLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class TopicCategory(str, Enum):
    PYTHON = "python"
    JAVA = "java"
    AWS = "aws"
    AI = "ai"
    MACHINE_LEARNING = "machine_learning"
    RAG = "rag"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    GIT = "git"
    SYSTEM_DESIGN = "system_design"
    SQL = "sql"
    DEVOPS = "devops"
    CYBERSECURITY = "cybersecurity"
    CLOUD = "cloud"
    GENERIC = "generic"


class VisualType(str, Enum):
    INFOGRAPHIC = "infographic"
    FLOWCHART = "flowchart"
    ARCHITECTURE = "architecture"
    PIPELINE = "pipeline"
    DATA_FLOW = "data_flow"
    CODE = "code"
    COMPARISON = "comparison"
    QUIZ = "quiz"
    CLUSTER_DIAGRAM = "cluster_diagram"
    CONTAINER_DIAGRAM = "container_diagram"
    BRANCH_DIAGRAM = "branch_diagram"
    QUERY_TABLE = "query_table"
    CI_CD_PIPELINE = "ci_cd_pipeline"
    ATTACK_DEFENSE_FLOW = "attack_defense_flow"
    SERVICE_ARCHITECTURE = "service_architecture"


class QuizOption(BaseModel):
    text: str
    is_correct: bool = False


class Quiz(BaseModel):
    question: str
    options: List[QuizOption] = Field(min_length=4, max_length=4)
    explanation: str


class FlowchartStep(BaseModel):
    label: str
    description: Optional[str] = None


class Flowchart(BaseModel):
    title: str
    steps: List[FlowchartStep] = Field(min_length=3, max_length=7)
    visual_type: VisualType = VisualType.FLOWCHART


class InfographicPoint(BaseModel):
    label: str
    value: str
    icon: Optional[str] = None


class Infographic(BaseModel):
    title: str
    points: List[InfographicPoint] = Field(min_length=3, max_length=6)
    visual_type: VisualType = VisualType.INFOGRAPHIC


class CodeSnippet(BaseModel):
    language: str
    title: str
    content: str
    highlight_lines: List[int] = []


class ArchitectureComponent(BaseModel):
    name: str
    description: str
    icon: Optional[str] = None
    connections: List[str] = []


class ArchitectureDiagram(BaseModel):
    title: str
    components: List[ArchitectureComponent] = Field(min_length=3, max_length=8)
    visual_type: VisualType = VisualType.ARCHITECTURE


class ComparisonRow(BaseModel):
    feature: str
    option_a: str
    option_b: str


class ComparisonTable(BaseModel):
    title: str
    header_a: str
    header_b: str
    rows: List[ComparisonRow] = Field(min_length=3, max_length=6)
    visual_type: VisualType = VisualType.COMPARISON


class EducationalContent(BaseModel):
    topic: str
    category: TopicCategory = TopicCategory.GENERIC
    audience: List[str] = ["students", "developers"]
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE

    hook: str

    infographic: Optional[Infographic] = None
    flowchart: Optional[Flowchart] = None
    architecture: Optional[ArchitectureDiagram] = None
    code: Optional[CodeSnippet] = None
    comparison: Optional[ComparisonTable] = None
    quiz: Optional[Quiz] = None

    takeaway: str
    cta: str = "Save this for later!"

    visual_strategy: List[VisualType] = Field(default_factory=list)


class RenderConfig(BaseModel):
    platform: Literal["facebook", "instagram"]
    width: int = 1080
    height: int = 1350
    theme_color: str = "#3B82F6"
    font_family: str = "Inter"
    brand_handle: str = "@Vijayakumarj_ai"
    logo_path: Optional[str] = None