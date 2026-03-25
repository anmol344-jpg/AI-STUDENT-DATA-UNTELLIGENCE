# decision_engine.py
# Decision Engine: rule-based actionable suggestions (AI Coach)
def generate_suggestions(features: dict):
    suggestions = []
    att = features.get("attendance", 0)
    study = features.get("study_hours", 0)
    marks = features.get("previous_marks", 0)
    assigns = features.get("assignments_completed", 0)
    sleep = features.get("sleep_hours", 7)

    if att < 60:
        suggestions.append("Attendance is very low — arrange meetings, set attendance goals, and enable reminders.")
    elif att < 75:
        suggestions.append("Attendance is below average — encourage class participation and schedule catch-up sessions.")

    if study < 2:
        suggestions.append("Study hours are low — start a daily 2–3 hour focused study routine using Pomodoro.")
    elif study < 4:
        suggestions.append("Consider increasing study time and structuring study plans per subject.")

    if marks < 50:
        suggestions.append("Previous marks are low — prioritize revision of fundamentals and use targeted tutoring.")
    elif marks < 65:
        suggestions.append("Marks are moderate — focus on weak topics and frequent self-testing.")

    if assigns < 5:
        suggestions.append("Low assignment completion — set incremental deadlines and peer accountability.")
    if sleep < 6:
        suggestions.append("Sleep is low — aim for 7–8 hours to improve cognitive performance.")
    if not features.get("extracurricular_activity", 0):
        suggestions.append("Encourage engagement in extracurriculars to improve wellbeing and engagement.")

    if not suggestions:
        suggestions.append("Student appears on track; maintain current habits and periodic checks.")
    return suggestions