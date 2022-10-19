import abydos.distance as abd
import gradio as gr


def company_comparator(company_1, company_2):
    if abd.DiscountedLevenshtein().sim(company_1, company_2) > 0.6:
        return "The company names are the same!"
    return "These are completely different companies."


demo = gr.Interface(
    fn=company_comparator,
    inputs=[
        gr.Textbox(placeholder="First Comany Name Here..."), 
        gr.Textbox(placeholder="Second Comany Name Here...")
    ],
    outputs="text",
)


demo.launch()
