from random import randint
import gradio as gr


def company_comparator(comany_1, company_2):
    if randint(0, 1) == 0:
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
