"""US AQI category bands, colors, and health messages — used by the dashboard."""

AQI_BANDS = [
    (0, 50, "Good", "#00e400", "Air quality is satisfactory."),
    (51, 100, "Moderate", "#ffff00", "Acceptable; unusually sensitive people should consider limiting prolonged exertion."),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00", "Sensitive groups may experience health effects."),
    (151, 200, "Unhealthy", "#ff0000", "Everyone may begin to experience health effects."),
    (201, 300, "Very Unhealthy", "#8f3f97", "Health alert: everyone may experience more serious effects."),
    (301, 500, "Hazardous", "#7e0023", "Health warning of emergency conditions."),
]


def categorize(aqi: float) -> tuple[str, str, str]:
    """Returns (label, color_hex, health_message) for a given AQI value."""
    if aqi is None:
        return "Unknown", "#9e9e9e", ""
    for lo, hi, label, color, msg in AQI_BANDS:
        if lo <= aqi <= hi:
            return label, color, msg
    if aqi > 500:
        return "Hazardous", "#7e0023", AQI_BANDS[-1][4]
    return "Unknown", "#9e9e9e", ""
