"""Generate a fictional sample UIC transcript PDF for demo screenshots.

Same structure as the user's real Graduation Audit Report but with a different
(fictional) student name + higher GPA. Output: data/sample_transcript_high_gpa.pdf

Run:
    python scripts/gen_sample_transcript.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ------------------------ Student data ------------------------
STUDENT = {
    "name_zh": "李明轩",
    "name_en": "LI Mingxuan",
    "no": "2330025088",
    "gender": "M",
    "award": "Bachelor of Science (Honours) in Artificial Intelligence",
    "admission": "September 2023",
    "gpa": "3.78",
    "total_gained": "110.0",
    "total_required": "148.0",
    "total_pending": "21.0",
    "lvl3_required": "36.0",
    "lvl3_gained": "12.0",
}


# Core/Major Required — same structure as real transcript, higher grades
CORE_DONE = [
    ("DS1013", "Python Programming for Beginners",      3.0, "A",  12.00),
    ("AI1023", "Database Management Systems",            3.0, "A-", 11.01),
    ("AI2003", "Data Structures and Algorithm Analysis", 3.0, "A",  12.00),
    ("AI2013", "Introduction to Artificial Intelligence", 3.0, "A-", 11.01),
    ("AI2023", "Artificial Intelligence Workshop",        3.0, "A",  12.00),
    ("AI2033", "Probability and Statistics",              3.0, "A-", 11.01),
    ("AI3003", "Neural Networks and Deep Learning",       3.0, "A",  12.00),
    ("AI3163", "Computer Architecture and Operating Systems", 3.0, "B+", 9.99),
    ("MATH2003", "Discrete Structures",                   3.0, "A-", 11.01),
]
CORE_CURRENT = ["AI1013", "AI3013", "AI3023", "AI4003", "COMP3023"]
CORE_CURRENT_TITLES = {
    "AI1013": "Object-Oriented Programming",
    "AI3013": "Machine Learning",
    "AI3023": "Machine Learning Workshop",
    "AI4003": "Optimization for Machine Learning",
    "COMP3023": "Design and Analysis of Algorithms",
}
CORE_TODO = ["AI3043", "AI4004", "MATH1003", "MATH1123"]
CORE_TODO_TITLES = {
    "AI3043": "Bayesian Networks",
    "AI4004": "Final Year Project I (AI)",
    "MATH1003": "Linear Algebra",
    "MATH1123": "Calculus for Science and Engineering",
}

# Major Elective — better progress than real transcript
MAJOR_ELEC_DONE = [
    ("COMP4243", "Mathematical and Computing Methods", 3.0, "A", 12.00),
    ("AI3133",   "Natural Language Processing",         3.0, "A-", 11.01),
]
MAJOR_ELEC_CURRENT = [
    ("COMP4153", "Quantum Finance and Intelligent Financial Trading Systems"),
    ("DS4083",   "Big Data Analytics"),
]
MAJOR_ELEC_TODO = [
    ("AI2053", "Introduction to Cognitive Science"),
    ("AI2063", "Game Theory"),
    ("AI3033", "Introduction to Robotics"),
    ("AI3053", "Intelligent Agent Technology"),
    ("AI3063", "Neuroscience in Artificial Intelligence"),
    ("AI3073", "Introduction to Bioinformatics"),
    ("AI3093", "Decision Theory"),
    ("AI3103", "Regression Analysis"),
    ("AI3113", "Speech Processing and Recognition"),
    ("AI3123", "Digital Image Processing"),
    ("AI3143", "Computer Vision"),
    ("AI3153", "Human-Computer Interaction"),
    ("AI4005", "Final Year Project II (AI)"),
    ("AI4013", "Knowledge Graph Engineering"),
    ("AI4023", "Deep Reinforcement Learning"),
    ("AI4033", "Large-Scale Distributed Multi-Agent Systems"),
    ("AI4053", "Fintech"),
    ("AI4063", "Pattern Recognition"),
    ("AI4083", "Multimedia Mining and Analytics"),
    ("AI4093", "Design and Implementation of Intelligent Vision System"),
    ("COMP3263", "Intelligent Internet of Things"),
    ("COMP3273", "5G Networks and Mobile Computing"),
    ("COMP4003", "Theory of Computation"),
    ("COMP4043", "Data Mining and Knowledge Discovery"),
    ("COMP4253", "AI-Generated Content"),
    ("DS4073", "Introduction to Data Visualisation"),
    ("DS4093", "Introduction to Recommender System"),
    ("MATH1153", "Applied Linear Algebra and Linear Dynamics"),
    ("MATH1163", "Advanced Calculus"),
    ("MATH3153", "Advanced Probability"),
    ("PHYS2003", "Principles of Physics"),
    ("STAT4013", "Multivariate Analysis"),
]

# University Core — graded courses + satisfactory pass-only
UNI_CORE_DONE = [
    ("UCLC1003", "University Chinese",                              3.0, "A-",  11.01),
    ("UCLC1013", "English for Academic Purposes I",                 3.0, "B+",  9.99),
    ("UCLC1023", "English for Academic Purposes II",                3.0, "A-",  11.01),
    ("UCLC1033", "English for Academic Purposes III",               3.0, "A-",  11.01),
    ("CHI1063",  "Chinese Culture and Modern China",                3.0, "A-",  11.01),
    ("CHI1073",  "Contemporary Chinese Society and Thought I",      3.0, "S",   0.00),
    ("CHI1103",  "Introduction to Modern Social Theories",          3.0, "S",   0.00),
    ("CHI1193",  "Contemporary World and China",                    2.0, "S",   0.00),
    ("CHI1203",  "Morality and Foundations of Law",                 3.0, "S",   0.00),
    ("CHI1253",  "Contemporary Chinese Society and Thought II",     3.0, "S",   0.00),
    ("MT1003",   "Military Training",                               2.0, "S",   0.00),
    ("WPEX2013", "Experiential Arts",                               1.0, "S",   0.00),
    ("WPEX1013", "Emotional Intelligence",                          1.0, "S",   0.00),
    ("WPEX2033", "Environmental Awareness",                         1.0, "S",   0.00),
    ("UCHL1123", "Tennis",                                          1.0, "A-",  3.67),
    ("UCHL1153", "Football",                                        1.0, "A-",  3.67),
    ("UCHL1243", "Golf",                                            1.0, "B+",  3.33),
]

# General Education
GE_DONE = [
    ("GFVM1023", "Applied Ethics: An Interdisciplinary Exploration", 3.0, "B+",  9.99),
    ("GFQR1003", "A Journey with Data",                              3.0, "A-",  11.01),
    ("GTCU2043", "Corporate Sustainability and Industry Innovation", 3.0, "A",   12.00),
    ("GTCU2123", "Exploring Culture and Music in Asia",              3.0, "A-",  11.01),
    ("GTSC2093", "IT for Success in Everyday Life and Work",         3.0, "A",   12.00),
]
GE_TODO = [
    ("GFHC----", "Foundational Courses - History and Civilization"),
    ("GCAP----", "GE Capstone Courses"),
]

# Free Elective
FREE_ELEC_DONE = [
    ("MATH1073", "Calculus I",                          3.0, "A",   12.00),
    ("ENG1013",  "Introduction to the Study of Language", 3.0, "A-", 11.01),
    ("ENG2113",  "Pragmatics and Discourse Analysis",    3.0, "B+",  9.99),
    ("ENG1003",  "Introduction to the Study of Literature", 3.0, "A-", 11.01),
    ("ENG2043",  "Speech and Oral Communications",       3.0, "B+",  9.99),
    ("ENG2223",  "English Through Media",                3.0, "A-",  11.01),
    ("COMP1023", "Foundations of C Programming",         3.0, "A",   12.00),
]


# ------------------------ PDF rendering ------------------------
def _style():
    base = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=base["Heading1"],
                           alignment=1, fontSize=14, spaceAfter=2, leading=18, fontName="Helvetica-Bold")
    sub = ParagraphStyle("sub", parent=base["Heading2"],
                         alignment=1, fontSize=14, spaceAfter=10, leading=18, fontName="Helvetica-Bold")
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontSize=11, spaceBefore=8, spaceAfter=4,
                        fontName="Helvetica-Bold")
    body = ParagraphStyle("body", parent=base["BodyText"], fontSize=9, leading=11)
    small = ParagraphStyle("small", parent=base["BodyText"], fontSize=8, leading=10, textColor=colors.grey)
    note = ParagraphStyle("note", parent=base["BodyText"], fontSize=8, leading=10, fontName="Helvetica-Oblique")
    return {"title": title, "sub": sub, "h2": h2, "body": body, "small": small, "note": note}


def _profile_table(stl):
    # Skip the (Chinese) name field — the default Helvetica font doesn't
    # render CJK and the LLM extractor doesn't read names anyway.
    data = [
        ["Student Name:", STUDENT["name_en"], "", ""],
        ["Student No.:", STUDENT["no"], "Gender:", STUDENT["gender"]],
        ["Award:", STUDENT["award"], "", ""],
        ["Year of Admission:", STUDENT["admission"], "", ""],
    ]
    t = Table(data, colWidths=[3.2*cm, 5.6*cm, 2.2*cm, 5.5*cm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
        ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _summary_box(stl):
    rows = [
        ["College General Requirements", "", "", ""],
        ["Cum. GPA Required", "Required:", ">= 2.00", "Satisfied"],
        ["", "Gained:", STUDENT["gpa"], ""],
        ["Programme Requirements", "", "", ""],
        ["Total Units", "Required:", STUDENT["total_required"], "Not Satisfied"],
        ["", "Gained:", f"{STUDENT['total_gained']}  (Excl. Repeated Courses)", ""],
        ["", "Pending:", STUDENT["total_pending"], ""],
        ["Units of Courses at Level 3 or above", "Required:", STUDENT["lvl3_required"], "Not Satisfied"],
        ["", "Gained:", STUDENT["lvl3_gained"], ""],
    ]
    t = Table(rows, colWidths=[7*cm, 2.5*cm, 4.5*cm, 2.5*cm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (0, 0), "Helvetica-Bold", 9.5),
        ("FONT", (0, 3), (0, 3), "Helvetica-Bold", 9.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _section_header(label, req, gained, stl):
    data = [[label, "Units Required:", str(req), "Units Gained:", str(gained)]]
    t = Table(data, colWidths=[8*cm, 3*cm, 1.5*cm, 3*cm, 1*cm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (0, 0), "Helvetica-Bold", 10),
        ("FONT", (1, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (2, 0), (2, 0), "Helvetica-Bold", 9.5),
        ("FONT", (4, 0), (4, 0), "Helvetica-Bold", 9.5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _course_table(rows, stl, *, with_grade=True, remark_for=None):
    header = ["Course Code", "Course Title", "Units", "Grade", "Grade Points Gained", "Remarks Code", "Other Info."]
    body = [header]
    for r in rows:
        if with_grade and len(r) == 5:
            code, title, units, grade, gp = r
            body.append([code, title, f"{units:.1f}", grade, f"{gp:.2f}", remark_for or "--", "--"])
        else:
            code, title = r[0], r[1]
            units = r[2] if len(r) >= 3 else 3.0
            body.append([code, title, f"{units:.1f}", "--", "--", "--", "--"])
    t = Table(body, colWidths=[2.4*cm, 6.6*cm, 1.1*cm, 1.1*cm, 2.4*cm, 1.6*cm, 1.4*cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Oblique", 8.5),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
        ("ALIGN", (2, 1), (4, -1), "CENTER"),
        ("ALIGN", (5, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, 0), 0.25, colors.black),
    ]))
    return t


def _subhead(label, stl):
    return Paragraph(f"<b><i>{label}</i></b>", stl["body"])


def _summary_table(stl):
    header = ["SUMMARY", "Units Required", "Units Gained", "Currently Enrolled", "Units Outstanding", "Units Satisfied"]
    rows = [
        header,
        ["1. CORE / MAJOR REQUIRED COURSES", "54.0", "27.0", "15.0", "12.0", "No"],
        ["2. MAJOR ELECTIVE COURSES",         "21.0",  "6.0",  "6.0",  "9.0", "No"],
        ["3. UNIVERSITY CORE COURSES",        "37.0", "37.0",  "0.0",  "0.0", "Yes"],
        ["4. GENERAL EDUCATION COURSE",       "18.0", "15.0",  "0.0",  "3.0", "No"],
        ["5. FREE ELECTIVE COURSES",          "18.0", "21.0",  "0.0",  "0.0", "Yes"],
        ["TOTAL UNITS",                       "148.0","106.0", "21.0", "24.0", ""],
        ["COURSE LEVEL SUMMARY", "", "", "", "", ""],
        ["6. AT LEVEL 3 OR ABOVE",            "36.0", "12.0", "18.0",  "6.0", "No"],
    ]
    t = Table(rows, colWidths=[6.8*cm, 2.4*cm, 2.2*cm, 2.6*cm, 2.6*cm, 2.4*cm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9.5),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 6), (-1, 6), "Helvetica-Bold", 9.5),
        ("FONT", (0, 7), (-1, 7), "Helvetica-Bold", 9.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
        ("LINEABOVE", (0, 6), (-1, 6), 0.4, colors.black),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def build():
    out = Path(__file__).resolve().parents[1] / "data" / "sample_transcript_high_gpa.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    stl = _style()
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=1.6*cm, bottomMargin=1.6*cm,
    )
    flow = []

    flow.append(Paragraph("Beijing Normal-Hong Kong Baptist University", stl["title"]))
    flow.append(Paragraph("Graduation Audit Report", stl["sub"]))
    flow.append(_profile_table(stl))
    flow.append(Spacer(1, 0.4*cm))
    flow.append(_summary_box(stl))
    flow.append(Spacer(1, 0.4*cm))

    # ===== A. Core/Major Required =====
    flow.append(_section_header("A. Core / Major Required Courses", "54.0", "27.0", stl))
    flow.append(_subhead("A1. Courses successfully completed:", stl))
    flow.append(_course_table(CORE_DONE, stl))
    flow.append(Spacer(1, 0.15*cm))
    flow.append(_subhead("A2. Courses currently taking:", stl))
    flow.append(_course_table(
        [(c, CORE_CURRENT_TITLES[c], 3.0) for c in CORE_CURRENT],
        stl, with_grade=False))
    flow.append(Spacer(1, 0.15*cm))
    flow.append(_subhead("A3. Courses failed/incomplete/to be taken:", stl))
    flow.append(_course_table(
        [(c, CORE_TODO_TITLES[c], 3.0) for c in CORE_TODO],
        stl, with_grade=False))
    flow.append(Spacer(1, 0.3*cm))

    # ===== B. Major Elective =====
    flow.append(_section_header("B. Major Elective Courses", "21.0", "6.0", stl))
    flow.append(_subhead("B1. Courses successfully completed:", stl))
    flow.append(_course_table(MAJOR_ELEC_DONE, stl))
    flow.append(Spacer(1, 0.15*cm))
    flow.append(_subhead("B2. Courses currently taking:", stl))
    flow.append(_course_table([(c, n, 3.0) for c, n in MAJOR_ELEC_CURRENT], stl, with_grade=False))
    flow.append(Spacer(1, 0.15*cm))
    flow.append(_subhead("B3. Courses failed/incomplete/to be selected:", stl))
    flow.append(_course_table([(c, n, 3.0) for c, n in MAJOR_ELEC_TODO], stl, with_grade=False))
    flow.append(Spacer(1, 0.3*cm))

    flow.append(PageBreak())

    # ===== C. University Core =====
    flow.append(_section_header("C. University Core Courses", "37.0", "37.0", stl))
    flow.append(_subhead("C1. Courses successfully completed:", stl))
    flow.append(_course_table(UNI_CORE_DONE, stl))
    flow.append(Spacer(1, 0.15*cm))
    flow.append(Paragraph("<i>C2. Courses currently taking: (NIL)</i>", stl["body"]))
    flow.append(Paragraph("<i>C3. Courses failed/incomplete/to be taken: (NIL)</i>", stl["body"]))
    flow.append(Spacer(1, 0.3*cm))

    # ===== D. General Education =====
    flow.append(_section_header("D. General Education Course", "18.0", "15.0", stl))
    flow.append(_subhead("D1. Courses successfully completed:", stl))
    flow.append(_course_table(GE_DONE, stl))
    flow.append(Spacer(1, 0.15*cm))
    flow.append(Paragraph("<i>D2. Courses currently taking: (NIL)</i>", stl["body"]))
    flow.append(_subhead("D3. Courses failed/incomplete/to be taken:", stl))
    flow.append(_course_table([(c, n, 3.0) for c, n in GE_TODO], stl, with_grade=False))
    flow.append(Spacer(1, 0.3*cm))

    # ===== E. Free Elective =====
    flow.append(_section_header("E. Free Elective Courses", "18.0", "21.0", stl))
    flow.append(_subhead("E1. Courses successfully completed:", stl))
    flow.append(_course_table(FREE_ELEC_DONE, stl))
    flow.append(Spacer(1, 0.15*cm))
    flow.append(Paragraph("<i>E2. Courses currently taking: (NIL)</i>", stl["body"]))
    flow.append(Paragraph("<i>E3. Courses failed/incomplete/to be taken: (NIL)</i>", stl["body"]))

    flow.append(PageBreak())

    # ===== Summary table =====
    flow.append(_summary_table(stl))
    flow.append(Spacer(1, 0.4*cm))
    flow.append(Paragraph("<b>Notes:</b>", stl["body"]))
    flow.append(Paragraph("(1) Units outstanding: excluding the currently enrolled units.", stl["small"]))
    flow.append(Paragraph("(2) The first digit of the number code signifies the level of the course.", stl["small"]))
    flow.append(Paragraph("(3) Please refer to your Undergraduate Handbook for graduation requirements.", stl["small"]))
    flow.append(Spacer(1, 0.6*cm))
    flow.append(Paragraph(
        "<i>This is a FICTIONAL sample transcript generated for SkillPath demo purposes. "
        "Student name, ID, grades and details are not real.</i>",
        stl["note"]))

    doc.build(flow)
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    build()
