"""Builds the PDF write-up of the 'Can an LLM replace training data?' experiment."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

HERE = Path(__file__).resolve().parent
RUN = HERE / "results" / "llm-20261122-183752"
OUT = HERE / "Knowledge_instead_of_data.pdf"

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=10.2, leading=14.5,
                      spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.6, leading=11.5, textColor=colors.HexColor("#444444"))
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14.5, spaceBefore=12,
                    spaceAfter=6, textColor=colors.HexColor("#1b2a41"), keepWithNext=1)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, spaceBefore=8,
                    spaceAfter=4, textColor=colors.HexColor("#1b2a41"), keepWithNext=1)
title = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=25,
                       textColor=colors.HexColor("#1b2a41"))
subtitle = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, fontSize=10.5, textColor=colors.HexColor("#555555"))
abstract = ParagraphStyle("abs", parent=body, leftIndent=18, rightIndent=18, fontSize=10, leading=14,
                          backColor=colors.HexColor("#f2f5f9"), borderPadding=(8, 8, 8, 8), spaceBefore=6,
                          spaceAfter=14)
cell = ParagraphStyle("cell", parent=body, fontSize=8.8, leading=11, spaceAfter=0)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")


def P(t, s=body):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(f"&bull;&nbsp;&nbsp;{i}", ParagraphStyle("b", parent=body, leftIndent=14,
                                                                firstLineIndent=-10, spaceAfter=3))
            for i in items]


def table(rows, widths, highlight=()):
    data = [[Paragraph(str(c), cellb if r == 0 else cell) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2a41")),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d1dc")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    for r in range(1, len(rows)):
        if r % 2 == 0:
            st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f6f8fb")))
    for r in highlight:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#e3f1e6")))
    # header text must be white: rebuild header paragraphs with white style
    hdr = ParagraphStyle("hdr", parent=cellb, textColor=colors.white)
    t._cellvalues[0] = [Paragraph(str(c), hdr) for c in rows[0]]
    t.setStyle(TableStyle(st))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(inch, 0.55 * inch, "Knowledge instead of data: LLM-written neural networks")
    canvas.drawRightString(letter[0] - inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


s = []
s += [Spacer(1, 6),
      P("Knowledge Instead of Data: Can an LLM Write a Working Neural Network Without Training Examples?", title),
      P("Independent research notebook &middot; weight-space programming &middot; November 2026", subtitle),
      Spacer(1, 8)]

s.append(P(
    "<b>Abstract.</b> Neural networks normally need labeled data, which is scarce in many real problems. "
    "We test whether a large language model (LLM) can replace that data by writing a network's weights directly "
    "from its own knowledge, an idea known as weight-space programming. An LLM (GPT-6 Astra via the Codex CLI, "
    "fresh session per attempt, no tools) wrote every weight of two small networks for two tasks: diagnosing "
    "breast tumors from 30 named measurements, and approving loans under a rule stated in plain English. It saw "
    "no training examples. We compared its networks against the same networks trained normally on 5 to 250 "
    "labeled examples. With zero data, the LLM's networks matched training on roughly 115 to 250+ examples: "
    "about 97% accuracy on tumors and about 94% on loans, where training on all 1,400 available examples reached "
    "only 87 to 90%. On 100,000 new loan applicants, and on 100,000 from a shifted population, the LLM's network "
    "stayed at about 95%, the best achievable given 5% label noise, while trained networks degraded under the "
    "shift. When input names were hidden, the LLM declined to guess, showing that its success came from "
    "understanding the inputs. Fine-tuning its network on a handful of examples made it worse. The main caveat "
    "is that the loan rule was given to the LLM, so that task measures translating a known rule into weights "
    "rather than discovering it.", abstract))

s.append(P("1. Motivation", h1))
s.append(P(
    "The goal behind this experiment is simple: <i>imagine a situation where we don't have enough data to train a "
    "neural network, and we use weight-space programming instead, because it does not require any data; it "
    "depends on another LLM to build the network.</i> If an LLM already knows how a task works, it might be able "
    "to write that knowledge straight into a network's weights, so that the network works before it has seen a "
    "single example."))
s.append(P(
    "An earlier experiment in this notebook, in which an LLM wrote the weights for a missing digit in an MNIST "
    "classifier, gave mixed results. There the network's internal features were invented by training and had no "
    "names, so the LLM had no way to connect its knowledge to them. This experiment fixes that: every input has a "
    "meaningful name the LLM can understand."))
s.append(P("<b>Research question.</b> An LLM-written network with <b>zero</b> examples performs as well as a "
           "normally trained network with <b>N</b> examples. What is N?"))

s.append(P("2. Setup", h1))
s.append(P("2.1 Tasks", h2))
s.append(table([
    ["Task", "Inputs", "Output", "Data", "Why it is used"],
    ["Tumor diagnosis", "30 measurements of cell nuclei (radius, texture, concavity, ...)", "malignant or benign",
     "569 real cases (scikit-learn); 398 pool, 171 test", "Real data; the LLM has medical knowledge"],
    ["Loan approval (invented)", "8 applicant features; 4 matter, 4 are distractors", "approved or rejected",
     "2,000 generated applicants; 1,400 pool, 600 test; 5% random label errors",
     "The LLM cannot have memorized it"],
], [1.05 * inch, 1.65 * inch, 0.85 * inch, 1.45 * inch, 1.5 * inch]))
s.append(Spacer(1, 5))
s.append(P("The loan rule given to the LLM: approve only if income-to-loan ratio &gt; 2.0, debt-to-income &lt; 0.40, "
           "years employed &ge; 2, and missed payments in the last 2 years &lt; 2. Age, dependents, savings and credit "
           "history play no role.", small))

s.append(P("2.2 Networks", h2))
s.append(table([
    ["Network", "Layers", "Tumor weights", "Loan weights"],
    ["Simple (logistic regression)", "inputs to 1 output neuron (sigmoid); no hidden layer", "31", "9"],
    ["Small (two-layer)", "inputs to 8 hidden neurons (ReLU) to 1 output neuron (sigmoid)", "257", "81"],
], [1.7 * inch, 2.9 * inch, 0.95 * inch, 0.95 * inch]))
s.append(Spacer(1, 4))
s.append(P("Inputs were standardized (0 = average) using means and spreads computed on the pool without labels; "
           "the LLM was told these means and spreads.", small))

s.append(P("3. Procedure", h1))
for head, text in [
    ("Phase 1, setup.", "A dedicated folder and Python environment (PyTorch, scikit-learn)."),
    ("Phase 2, tasks.", "Both datasets prepared; a locked test set split off that no method ever trains on."),
    ("Phase 3, training baselines.", "Both networks trained normally (Adam, 300 steps, light L2) on 5, 10, 25, 50, "
     "100 and 250 labeled examples, 20 random draws each; random-weight networks scored for reference."),
    ("Phase 4, pilot.", "One LLM call to verify the pipeline end to end."),
    ("Phase 5, LLM runs.", "For each task, Codex wrote (a) the simple network's weights, (b) the small network's "
     "weights, (c) a Python rule instead of weights, and (d) the simple network's weights with input names hidden "
     "(x1, x2, ...). Each attempt was a fresh, ephemeral session with an identical prompt, no training data, and no "
     "tools; tool use was logged and any run that used tools would be discarded. The reply's numbers were loaded "
     "directly into the network as PyTorch tensors, with no training applied."),
    ("Phase 6, starting point.", "The LLM's weights used as the initialization for a little training (5 to 50 "
     "examples), compared with training from random initialization on the same examples."),
    ("Phase 7, generalization check.", "Every network re-scored on 100,000 new loan applicants, 100,000 applicants "
     "from a shifted population (richer, older, more debt; same rule), and all 569 tumor cases."),
]:
    s.append(P(f"<b>{head}</b> {text}"))
s.append(P("Of 40 planned LLM sessions, 30 completed; the remaining 10 (mostly the fifth repeat of each condition) "
           "were skipped when the Codex usage limit was reached. Skipped sessions were not counted. No completed "
           "session used tools.", small))

s.append(P("4. Results", h1))
s.append(P("4.1 Zero-data LLM networks versus training (original test sets)", h2))
s.append(table([
    ["Task", "Method", "Test accuracy", "AUROC", "Worth about N examples", "Runs"],
    ["Tumor", "Random weights", "52.9%", "0.540", "-", "20"],
    ["Tumor", "Training, 5 / 25 / 100 / 250 examples", "89.6 / 92.5 / 95.2 / 97.7%", "0.967-0.998", "-", "20 each"],
    ["Tumor", "LLM weights, simple network", "97.1% (96.5-98.2)", "0.997", "about 210", "4"],
    ["Tumor", "LLM weights, small network", "95.3% (93.6-96.5)", "0.992", "about 135", "4"],
    ["Tumor", "LLM Python rule", "95.5% (94.2-96.5)", "0.992", "about 115", "4"],
    ["Tumor", "LLM weights, names hidden", "63.7% (all zeros)", "0.500", "none", "4"],
    ["Loan", "Random weights", "45.3%", "0.432", "-", "20"],
    ["Loan", "Training small net, 5 / 25 / 100 / 250 examples", "60.5 / 65.3 / 74.1 / 79.4%", "0.599-0.825", "-",
     "20 each"],
    ["Loan", "Training small net, all 1,400 examples", "87.2%", "0.898", "-", "1"],
    ["Loan", "LLM weights, simple network", "71.4% (69.7-73.7)", "0.771", "about 200*", "4"],
    ["Loan", "LLM weights, small network", "94.0% (93.8-94.2)", "0.919", "more than 250", "4"],
    ["Loan", "LLM Python rule", "94.2%", "0.923", "more than 250", "3"],
    ["Loan", "LLM weights, names hidden", "65.8% (all zeros)", "0.500", "none", "3"],
], [0.6 * inch, 2.0 * inch, 1.45 * inch, 0.9 * inch, 0.95 * inch, 0.6 * inch], highlight=(3, 11, 12)))
s.append(Spacer(1, 4))
s.append(P("* The simple network cannot represent the loan rule's AND conditions; training it on all 1,400 examples "
           "reaches only 71.5%, so the LLM matched this network's practical ceiling. AUROC measures how well a network "
           "ranks positives above negatives (0.5 = chance, 1.0 = perfect).", small))

s.append(KeepTogether([Spacer(1, 6), Image(str(RUN / "chart_loan.png"), width=5.6 * inch, height=3.36 * inch),
                       P("Figure 1. Loan task: training accuracy rises with more examples (solid lines); the "
                         "LLM's zero-data networks are flat lines.", small)]))
s.append(KeepTogether([Image(str(RUN / "chart_tumor.png"), width=5.6 * inch, height=3.36 * inch),
                       P("Figure 2. Tumor task. Training with only 5 examples already reaches about 90%, so the "
                         "room between no data and full data is small on this task.", small)]))

s.append(P("4.2 Generalization on large and shifted test sets", h2))
s.append(table([
    ["Test set", "Always guess the common answer", "Trained small net, 250 ex.", "Trained small net, all ex.",
     "LLM small net, 0 ex."],
    ["Loan: 100,000 new applicants", "68.8%", "80.8%", "90.3%", "94.8%"],
    ["Loan: 100,000 shifted population", "74.4%", "79.3%", "90.0%", "94.9%"],
    ["Tumor: all 569 cases", "62.7%", "98.3%", "98.9%**", "96.6%"],
], [1.95 * inch, 1.25 * inch, 1.15 * inch, 1.15 * inch, 1.05 * inch], highlight=(1, 2)))
s.append(Spacer(1, 4))
s.append(P("With 5% of loan labels randomly flipped, about 95% is the best any method can score. The LLM's simple "
           "tumor network scored 97.5% on all 569 cases. ** Trained networks had seen some of these 569 cases during "
           "training, which favors them.", small))

s.append(P("4.3 Using the LLM's weights as a starting point", h2))
s.append(table([
    ["Examples used for training", "5", "10", "25", "50"],
    ["Loan small net, start from LLM weights", "67.6%", "70.1%", "79.3%", "79.8%"],
    ["Loan small net, start from random", "60.7%", "63.8%", "66.8%", "68.6%"],
    ["Tumor simple net, start from LLM weights", "86.7%", "88.9%", "91.9%", "93.5%"],
    ["Tumor simple net, start from random", "88.6%", "90.0%", "92.5%", "93.8%"],
], [2.6 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch, 0.85 * inch]))
s.append(Spacer(1, 4))
s.append(P("The LLM start helped on the loan task, but every fine-tuned network scored below the untouched LLM "
           "network (94.0%): a handful of examples pulled it away from the correct rule. On tumors it gave no benefit.",
           small))

s.append(P("5. Interpretation", h1))
s += bullets([
    "<b>An LLM can stand in for data when it understands the task.</b> With zero examples, its networks were worth "
    "roughly 115 to 250+ labeled examples, and on the loan task they beat training on all 1,400 examples.",
    "<b>Its networks generalize.</b> Accuracy held at about 95% on 100,000 unseen applicants and on a shifted "
    "population, where trained networks lost accuracy. The LLM encoded the rule itself, not quirks of one sample.",
    "<b>Meaningful input names are essential.</b> With names hidden, the LLM wrote all-zero weights and stated that it "
    "had no basis to guess. This also argues against memorized column orders.",
    "<b>Hidden layers are within reach.</b> It wrote a working two-layer network (8 hidden neurons) that expressed "
    "the loan rule's AND logic, which the simple network cannot.",
    "<b>Weights matched code.</b> Writing weights and writing a Python rule gave the same accuracy here.",
    "<b>Do not fine-tune on tiny data.</b> A few examples degraded the LLM's correct network.",
])

s.append(P("6. Limitations", h1))
s += bullets([
    "<b>The loan rule was given in words.</b> That task measures translating a stated rule into weights, not "
    "discovering a rule. It shows what is possible when an expert can describe the rule.",
    "<b>The tumor dataset is famous.</b> General familiarity may have helped, although the hidden-names control argues "
    "against memorized details. The tumor task is also easy: 5 examples already give about 90%.",
    "<b>Small scale.</b> 3 to 4 sessions per condition, one LLM at medium reasoning effort, tiny networks.",
    "<b>Input statistics were provided.</b> The LLM was told each feature's mean and spread (computed without labels).",
])

s.append(P("7. Conclusion and next step", h1))
s.append(P(
    "When a task can be described in words and its inputs have meaningful names, an LLM can write a neural network "
    "without any training data that performs as well as, or better than, one trained on hundreds of examples, and that "
    "generalizes to new and shifted populations. This supports the motivating idea for data-scarce settings. The decisive "
    "next test is a task where the LLM must rely on its own knowledge, no rule is stated, and the dataset is not famous; "
    "for example, a lesser-known real dataset with named features, again comparing against training at each data size."))
s.append(Spacer(1, 6))
s.append(P("Code and logs: experiments/knowledge_vs_data/ (run.py, tasks.py, generalize.py; per-session prompts, "
           "replies and weights in results/llm-20261122-183752/).", small))

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=inch, rightMargin=inch, topMargin=0.85 * inch,
                        bottomMargin=0.85 * inch, title="Knowledge Instead of Data",
                        author="IndependentR1", subject="LLM-written neural networks without training data")
doc.build(s, onFirstPage=footer, onLaterPages=footer)
print(OUT)
