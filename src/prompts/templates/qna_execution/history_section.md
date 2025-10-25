{% if history %}
Here are the most recent {{ history|length }} steps (showing last {{ history|length }} of all attempts):

{% for call in history %}
**Step {{ loop.index }}:**
{{ call.content }}

Result: {{ call.result_or_guidance }}

{% endfor %}

If the previous attempt failed or guidance indicates an issue, correct the plan accordingly and retry in this turn.
{% endif %}
