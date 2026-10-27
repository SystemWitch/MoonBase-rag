"""Create MoonBase handbook, finance, and engineering content in one PDF."""
from pathlib import Path
import fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# A simple embedded airlock illustration gives the PDF a real image to extract.
diagram = Image.new("RGB", (600, 300), "white")
d = ImageDraw.Draw(diagram)
d.text((20, 15), "AIRLOCK A - SIMPLIFIED PLAN", fill="black")
d.rectangle((70, 70, 530, 245), outline="#2878b5", width=5)
d.rectangle((250, 70, 350, 245), outline="#d55e00", width=4)
d.text((95, 140), "HABITAT", fill="black")
d.text((263, 125), "INNER HATCH", fill="black")
d.text((390, 140), "AIRLOCK", fill="black")
d.text((20, 270), "Illustrative diagram. Follow the runbook.", fill="black")
diagram_path = DATA / "airlock-diagram.png"
diagram.save(diagram_path)

doc = fitz.open()

# Handbook: all staff may retrieve this page.
p = doc.new_page()
p.insert_text((55, 42), "[ACCESS: all-staff]", fontsize=9)
p.insert_text((55, 75), "MoonBase Staff Handbook", fontsize=20)
p.insert_textbox(fitz.Rect(55, 105, 535, 190),
    "MoonBase is a shared lunar research station. Everyone must follow posted safety signs and report hazards to mission control. If an oxygen alarm sounds, move to the nearest marked safe zone and notify the on-duty engineer. Do not enter an airlock unless trained and authorized.", fontsize=12)
p.insert_text((55, 220), "Handbook quick reference", fontsize=14)
rows = [("Situation", "Everyone should"), ("Oxygen alarm", "Go to safe zone; alert engineer"), ("Airlock use", "Only trained, authorized crew")]
for r, row in enumerate(rows):
    for c, value in enumerate(row):
        rect = fitz.Rect(55 + c * 240, 245 + r * 34, 295 + c * 240, 279 + r * 34)
        p.draw_rect(rect, color=(0.2, 0.2, 0.2), fill=(0.88, 0.94, 0.98) if r == 0 else (1, 1, 1))
        p.insert_textbox(fitz.Rect(rect.x0 + 7, rect.y0 + 8, rect.x1 - 5, rect.y1 - 3), value, fontsize=9)

# Finance: salary information is restricted to the finance group.
p = doc.new_page()
p.insert_text((55, 42), "[ACCESS: finance]", fontsize=9)
p.insert_text((55, 75), "MoonBase Compensation Schedule - CONFIDENTIAL", fontsize=18)
p.insert_textbox(fitz.Rect(55, 105, 535, 155), "Annual base salaries are confidential finance records. Do not share outside the Finance group.", fontsize=12)
rows = [("Role", "Annual salary (USD)"), ("Flight engineer", "$142,000"), ("Research scientist", "$128,000"), ("Mission commander", "$165,000")]
for r, row in enumerate(rows):
    for c, value in enumerate(row):
        rect = fitz.Rect(55 + c * 240, 185 + r * 38, 295 + c * 240, 223 + r * 38)
        p.draw_rect(rect, color=(0.2, 0.2, 0.2), fill=(0.98, 0.9, 0.82) if r == 0 else (1, 1, 1))
        p.insert_text((rect.x0 + 8, rect.y0 + 24), value, fontsize=10)

# Engineering: detailed oxygen procedure and airlock diagram are engineer-only.
p = doc.new_page()
p.insert_text((55, 42), "[ACCESS: engineers]", fontsize=9)
p.insert_text((55, 75), "Oxygen Alarm Runbook", fontsize=20)
p.insert_textbox(fitz.Rect(55, 105, 535, 220),
    "For trained engineering crew. At an oxygen partial pressure below 19.5 kPa, acknowledge the alarm at the habitat console, check the redundant sensor, notify mission control, and inspect the life-support loop from the safe side of the inner hatch. Do not cycle the airlock during an active pressure fault. If pressure continues to fall, isolate the affected habitat section and follow mission control instructions.", fontsize=11)
p.insert_text((55, 245), "Airlock A diagram: habitat is on the left; the inner hatch separates it from the airlock on the right.", fontsize=10)
p.insert_image(fitz.Rect(55, 270, 455, 455), filename=str(diagram_path))

doc.set_metadata({"title": "MoonBase RAG Demo Documents"})
doc.save(DATA / "sample.pdf")
print(f"Created {DATA / 'sample.pdf'}")
