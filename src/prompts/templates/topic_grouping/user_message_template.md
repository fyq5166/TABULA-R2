Group these {{ domain }} tables into 2-10 table research topics using Title_Case_Naming.

GOOD EXAMPLES: {{ examples }}

RULES:
1. Create meaningful research groups (avoid single-table groups)
2. Use Title_Case format: "Core_Topic_and_Related_Aspect"
3. Group complementary datasets that enable cross-table analysis
4. Match titles exactly as given

TABLES ({{ titles|length }} total):
{% for title in titles -%}
{{ loop.index }}. {{ title }}
{% endfor %}

OUTPUT JSON:
{% raw %}
[
  {"group_name": "Topic_Name", "titles": ["exact_title_1", "exact_title_2"]},
  {"group_name": "Another_Topic", "titles": ["exact_title_3", "exact_title_4"]}
]
{% endraw %}

Return exactly this JSON array. Do not include any commentary, explanations, Markdown fences, or text outside the JSON structure.

JSON:
