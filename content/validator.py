from typing import List, Tuple, Optional
from content.schemas import EducationalContent, VisualType, Quiz, QuizOption


class ValidationError(Exception):
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_content(content: EducationalContent) -> Tuple[bool, List[str]]:
    """Validate educational content meets quality standards."""
    errors = []
    warnings = []
    
    # Basic required fields
    if not content.topic or len(content.topic.strip()) < 3:
        errors.append("Topic must be at least 3 characters")
    
    if not content.hook or len(content.hook.strip()) < 10:
        errors.append("Hook must be at least 10 characters")
    
    if not content.takeaway or len(content.takeaway.strip()) < 10:
        errors.append("Takeaway must be at least 10 characters")
    
    # Hook length check (should be readable in seconds)
    if content.hook and len(content.hook) > 200:
        warnings.append(f"Hook is long ({len(content.hook)} chars), may not read quickly on mobile")
    
    # Validate visual strategy matches content
    if content.visual_strategy:
        has_infographic = VisualType.INFOGRAPHIC in content.visual_strategy
        has_flowchart = VisualType.FLOWCHART in content.visual_strategy
        has_architecture = VisualType.ARCHITECTURE in content.visual_strategy
        has_code = VisualType.CODE in content.visual_strategy
        has_comparison = VisualType.COMPARISON in content.visual_strategy
        has_quiz = VisualType.QUIZ in content.visual_strategy
        
        if has_infographic and not content.infographic:
            errors.append("Visual strategy includes infographic but no infographic content provided")
        if has_flowchart and not content.flowchart:
            errors.append("Visual strategy includes flowchart but no flowchart content provided")
        if has_architecture and not content.architecture:
            errors.append("Visual strategy includes architecture but no architecture content provided")
        if has_code and not content.code:
            errors.append("Visual strategy includes code but no code content provided")
        if has_comparison and not content.comparison:
            errors.append("Visual strategy includes comparison but no comparison content provided")
        if has_quiz and not content.quiz:
            errors.append("Visual strategy includes quiz but no quiz content provided")
    
    # Validate quiz
    if content.quiz:
        quiz_errors = _validate_quiz(content.quiz)
        errors.extend(quiz_errors)
    
    # Validate flowchart
    if content.flowchart:
        fc_errors = _validate_flowchart(content.flowchart)
        errors.extend(fc_errors)
    
    # Validate infographic
    if content.infographic:
        ig_errors = _validate_infographic(content.infographic)
        errors.extend(ig_errors)
    
    # Validate architecture
    if content.architecture:
        arch_errors = _validate_architecture(content.architecture)
        errors.extend(arch_errors)
    
    # Validate code
    if content.code:
        code_errors = _validate_code(content.code)
        warnings.extend(code_errors)
    
    # Validate comparison
    if content.comparison:
        comp_errors = _validate_comparison(content.comparison)
        errors.extend(comp_errors)
    
    # Content balance check
    visual_count = sum([
        content.infographic is not None,
        content.flowchart is not None,
        content.architecture is not None,
        content.code is not None,
        content.comparison is not None,
    ])
    
    if visual_count == 0:
        warnings.append("No visual components provided - content may be text-heavy")
    elif visual_count > 4:
        warnings.append(f"Many visual components ({visual_count}) - may be cluttered on mobile")
    
    return len(errors) == 0, errors + warnings


def _validate_quiz(quiz: Quiz) -> List[str]:
    errors = []
    
    if not quiz.question or len(quiz.question.strip()) < 5:
        errors.append("Quiz question must be at least 5 characters")
    
    if len(quiz.options) != 4:
        errors.append(f"Quiz must have exactly 4 options, got {len(quiz.options)}")
    else:
        correct_count = sum(1 for opt in quiz.options if opt.is_correct)
        if correct_count != 1:
            errors.append(f"Quiz must have exactly 1 correct answer, got {correct_count}")
        
        for i, opt in enumerate(quiz.options):
            if not opt.text or len(opt.text.strip()) < 2:
                errors.append(f"Quiz option {i+1} must have text")
    
    if not quiz.explanation or len(quiz.explanation.strip()) < 10:
        errors.append("Quiz explanation must be at least 10 characters")
    
    return errors


def _validate_flowchart(fc) -> List[str]:
    errors = []
    
    if not fc.steps or len(fc.steps) < 3:
        errors.append("Flowchart must have at least 3 steps")
    elif len(fc.steps) > 7:
        errors.append(f"Flowchart has {len(fc.steps)} steps, max recommended is 7 for mobile")
    
    for i, step in enumerate(fc.steps):
        if not step.label or len(step.label.strip()) < 2:
            errors.append(f"Flowchart step {i+1} must have a label")
        if len(step.label) > 50:
            errors.append(f"Flowchart step {i+1} label is too long (>50 chars)")
    
    return errors


def _validate_infographic(ig) -> List[str]:
    errors = []
    
    if not ig.points or len(ig.points) < 3:
        errors.append("Infographic must have at least 3 points")
    elif len(ig.points) > 6:
        errors.append(f"Infographic has {len(ig.points)} points, max recommended is 6")
    
    for i, point in enumerate(ig.points):
        if not point.label or len(point.label.strip()) < 2:
            errors.append(f"Infographic point {i+1} must have a label")
        if not point.value or len(point.value.strip()) < 2:
            errors.append(f"Infographic point {i+1} must have a value")
        if len(point.label) > 30:
            errors.append(f"Infographic point {i+1} label is too long (>30 chars)")
        if len(point.value) > 80:
            errors.append(f"Infographic point {i+1} value is too long (>80 chars)")
    
    return errors


def _validate_architecture(arch) -> List[str]:
    errors = []
    
    if not arch.components or len(arch.components) < 3:
        errors.append("Architecture must have at least 3 components")
    elif len(arch.components) > 8:
        errors.append(f"Architecture has {len(arch.components)} components, max recommended is 8")
    
    for i, comp in enumerate(arch.components):
        if not comp.name or len(comp.name.strip()) < 2:
            errors.append(f"Architecture component {i+1} must have a name")
        if not comp.description or len(comp.description.strip()) < 5:
            errors.append(f"Architecture component {i+1} must have a description")
    
    return errors


def _validate_code(code) -> List[str]:
    warnings = []
    
    if not code.content or len(code.content.strip()) < 10:
        warnings.append("Code snippet seems very short")
    elif len(code.content) > 500:
        warnings.append(f"Code snippet is long ({len(code.content)} chars), may not fit on mobile")
    
    if not code.language:
        warnings.append("Code snippet missing language")
    
    # Check for obvious syntax issues
    lines = code.content.strip().split('\n')
    if len(lines) > 15:
        warnings.append(f"Code has {len(lines)} lines, consider shortening for mobile")
    
    return warnings


def _validate_comparison(comp) -> List[str]:
    errors = []
    
    if not comp.rows or len(comp.rows) < 3:
        errors.append("Comparison must have at least 3 rows")
    elif len(comp.rows) > 6:
        errors.append(f"Comparison has {len(comp.rows)} rows, max recommended is 6")
    
    for i, row in enumerate(comp.rows):
        if not row.feature or len(row.feature.strip()) < 2:
            errors.append(f"Comparison row {i+1} must have a feature")
        if not row.option_a:
            errors.append(f"Comparison row {i+1} missing option A")
        if not row.option_b:
            errors.append(f"Comparison row {i+1} missing option B")
    
    return errors


def auto_fix_content(content: EducationalContent) -> EducationalContent:
    """Attempt to auto-fix common issues."""
    import copy
    fixed = copy.deepcopy(content)
    
    # Fix quiz: ensure exactly one correct answer
    if fixed.quiz:
        correct_indices = [i for i, opt in enumerate(fixed.quiz.options) if opt.is_correct]
        if len(correct_indices) == 0 and fixed.quiz.options:
            fixed.quiz.options[0].is_correct = True
        elif len(correct_indices) > 1:
            for i, opt in enumerate(fixed.quiz.options):
                opt.is_correct = (i == correct_indices[0])
    
    # Trim overly long text
    if fixed.hook and len(fixed.hook) > 180:
        fixed.hook = fixed.hook[:177] + "..."
    
    if fixed.takeaway and len(fixed.takeaway) > 160:
        fixed.takeaway = fixed.takeaway[:157] + "..."
    
    # Trim flowchart step labels
    if fixed.flowchart:
        for step in fixed.flowchart.steps:
            if len(step.label) > 45:
                step.label = step.label[:42] + "..."
            if step.description and len(step.description) > 60:
                step.description = step.description[:57] + "..."
    
    # Trim infographic points
    if fixed.infographic:
        for point in fixed.infographic.points:
            if len(point.label) > 25:
                point.label = point.label[:22] + "..."
            if len(point.value) > 70:
                point.value = point.value[:67] + "..."
    
    return fixed