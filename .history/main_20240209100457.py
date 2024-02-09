import streamlit as st
from lida import Manager, TextGenerationConfig, llm
from lida.datamodel import Goal
import os
import pandas as pd
import base64

# make data dir if it doesn't exist
os.makedirs("data", exist_ok=True)

st.set_page_config(
    page_title="Education Databot: Automatic Generation of Visualizations and Infographics",
    page_icon="./static/unesco-16-168843.png",
)

# Function to get base64 string


def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


# Convert your image to base64 string
logo_base64 = get_image_base64("./static/UNESCO_UIS_logo_color_eng.jpg")

# Embed the base64 string directly in the HTML
st.markdown(
    f"""
    <div style='text-align: right;'>
        <img src="data:image/jpeg;base64,{logo_base64}" style='max-width: 150px; height: auto;'>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("# Education Databot: An AI-enhanced data visualization tool using SDG 4 data")

st.sidebar.write("## Setup")

# Step 1 - Get OpenAI API key
openai_key = os.getenv("AZURE_OPENAI_API_KEY")

if not openai_key:
    openai_key = st.sidebar.text_input("Enter OpenAI API key:")
    if openai_key:
        display_key = openai_key[:2] + "*" * \
            (len(openai_key) - 5) + openai_key[-2:]
        st.sidebar.write(f"Current key: {display_key}")
    else:
        st.sidebar.write("Please enter OpenAI API key.")
else:
    display_key = openai_key[:2] + "*" * \
        (len(openai_key) - 5) + openai_key[-3:]
    st.sidebar.write(
        f"Azure OpenAI API key loaded from environment variable: {display_key}")

st.markdown(
    """
    This prototype version of the Education Databot is designed to harness the power of machine learning to better visualize education data. The next development of the tool will include an interactive interface powered by an advanced chatbot. 
    Fed by the reliable database from the UNESCO Institute for Statistics for SDG 4, it provides a unique and valuable resource for informed decision-making.

    Education Databot facilitates the creation of data visualizations and infographics; supports a variety of programming and visualization resources, ensuring compatibility and ease of use. The Education Databot provides accurate and relevant information, ensuring the generated visualizations are both meaningful and reliable. Users can set objectives, choose goals, create tailored plots and can adjust visualizations for specific needs.

    **User guide**

    1. **Define Your Objectives:** begin by setting clear objectives for what you wish to achieve with your data analysis.
    2. **Select the dataset:** choose the dataset from the dropdown list or select your own. The tool will suggest Goals (questions) and display a chart.
    3. To customize your analysis, you can:
        - **Choose a Visualization Library:** pick from existing solutions depending on your preference and the nature of your data.
        - **Select Generated Goals:** browse through Education Databot suggested goals based on your dataset and select the ones that align with your objectives.
        - **Generate and Customize Visualizations:** use Education Databot’s generated code to create visualizations. You can adjust the code to suit your specific analytic needs.
    
    See the project page [here](https://github.com/roy-saurabh/edudatabot/blob/main/README.md) for more information.

    <span style='color:grey;'><strong><em>Disclaimer:</em></strong>
    <em>The Education Databot aims to deliver insightful education data visualizations, yet users should be aware of its prototype limitations. The tool's Retrieval-Augmented Generation (RAG) depends on a limited initial vector datastore, which may restrict data point access for visualization. Additionally, the choice of visual representation by our advanced libraries may not always align perfectly with the intended goals due to their developmental stages. Our AI models, including OpenAI's GPT-3.5 Turbo and Azure OpenAI API, along with libraries like LlamaIndex, Lida, Trulens, and AffectLog, are crafted to enhance visualization with UNESCO Institute for Statistics (UIS) data. However, their outputs are shaped by their training and algorithms, meaning visualizations should aid, not replace, comprehensive data analysis. We're actively working to broaden our data access, refine AI models, and improve visualization options. Feedback is welcome to enhance the tool's precision and utility. For optimal use, please refer to the project README for customization guidance and stay updated on our progress.</em>
    </span>
    
   ----
""",
    unsafe_allow_html=True
)

# Step 2 - Select a dataset and summarization method
if openai_key:
    # Initialize selected_dataset to None
    selected_dataset = None

    # select model from gpt-4 , gpt-3.5-turbo, gpt-3.5-turbo-16k
    st.sidebar.write("## Text Generation Model")
    models = ["gpt-3.5-turbo"]
    selected_model = st.sidebar.selectbox(
        'Choose a model',
        options=models,
        index=0
    )

    # select temperature on a scale of 0.0 to 1.0
    # st.sidebar.write("## Text Generation Temperature")
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0)

    # set use_cache in sidebar
    use_cache = st.sidebar.checkbox("Use cache", value=True)

    # Handle dataset selection and upload
    st.sidebar.write("## Data Summarization")
    st.sidebar.write("### Choose a dataset")

    datasets = [
        {"label": "Select a dataset", "url": None},
        {"label": "Completion rate","url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Completion_rate.csv"},
        {"label": "Education facilities and safety","url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Education_facilities_and_safety.csv"},
        {"label": "Equal access",
            "url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Equal_access.csv"},
        {"label": "Government expenditure","url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Government_expenditure.csv"},
        {"label": "Literacy and numeracy",
            "url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Literacy_and_numeracy.csv"},
        {"label": "Out of school","url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Out_of_school.csv"},
        {"label": "Pre-primary education",
            "url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Pre_primary_education.csv"},
        {"label": "Primary and secondary education",
            "url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Primary_and_secondary_education.csv"},
        {"label": "Scholarships",
            "url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Scholarships.csv"},
        {"label": "Skills", "url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Skills.csv"},
        {"label": "Sustainable development knowledge",
            "url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Sustainable_development_knowledge.csv"},
        {"label": "Teachers", "url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Teachers.csv"},
        {"label": "Technical vocational and tertiary",
            "url": "https://raw.githubusercontent.com/roy-saurabh/un-vision-ui/main/datasets/Technical_vocational_and_tertiary.csv"},
    ]

    selected_dataset_label = st.sidebar.selectbox(
        'Choose a dataset',
        options=[dataset["label"] for dataset in datasets],
        index=0
    )

    upload_own_data = st.sidebar.checkbox("Upload your own data")

    if upload_own_data:
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or JSON file", type=["csv", "json"])

        if uploaded_file is not None:
            # Get the original file name and extension
            file_name, file_extension = os.path.splitext(uploaded_file.name)

            # Load the data depending on the file type
            if file_extension.lower() == ".csv":
                data = pd.read_csv(uploaded_file)
            elif file_extension.lower() == ".json":
                data = pd.read_json(uploaded_file)

            # Save the data using the original file name in the data dir
            uploaded_file_path = os.path.join("data", uploaded_file.name)
            data.to_csv(uploaded_file_path, index=False)

            selected_dataset = uploaded_file_path

            datasets.append({"label": file_name, "url": uploaded_file_path})

            # st.sidebar.write("Uploaded file path: ", uploaded_file_path)
    else:
        selected_dataset = datasets[[dataset["label"]
                                     for dataset in datasets].index(selected_dataset_label)]["url"]

    if not selected_dataset:
        st.info(
            "To continue, select a dataset from the sidebar on the left or upload your own.")

    st.sidebar.write("### Choose a summarization method")
    # summarization_methods = ["default", "llm", "columns"]
    summarization_methods = [
        {"label": "llm",
         "description":
         "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},
        {"label": "default",
         "description": "Uses dataset column statistics and column names as the summary"},

        {"label": "columns", "description": "Uses the dataset column names as the summary"}]

    # selected_method = st.sidebar.selectbox("Choose a method", options=summarization_methods)
    selected_method_label = st.sidebar.selectbox(
        'Choose a method',
        options=[method["label"] for method in summarization_methods],
        index=0
    )

    selected_method = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["label"]

    # add description of selected method in very small font to sidebar
    selected_summary_method_description = summarization_methods[[
        method["label"] for method in summarization_methods].index(selected_method_label)]["description"]

    if selected_method:
        st.sidebar.markdown(
            f"<span> {selected_summary_method_description} </span>",
            unsafe_allow_html=True)

# Step 3 - Generate data summary
if openai_key and selected_dataset and selected_method:
    lida = Manager(text_gen=llm("openai", api_key=openai_key))
    textgen_config = TextGenerationConfig(
        n=1,
        temperature=temperature,
        model=selected_model,
        use_cache=use_cache)

    st.write("## Summary")
    # **** lida.summarize *****
    summary = lida.summarize(
        selected_dataset,
        summary_method=selected_method,
        textgen_config=textgen_config)

    if "dataset_description" in summary:
        st.write(summary["dataset_description"])

    if "fields" in summary:
        fields = summary["fields"]
        nfields = []
        for field in fields:
            flatted_fields = {}
            flatted_fields["column"] = field["column"]
            # flatted_fields["dtype"] = field["dtype"]
            for row in field["properties"].keys():
                if row != "samples":
                    flatted_fields[row] = field["properties"][row]
                else:
                    flatted_fields[row] = str(field["properties"][row])
            # flatted_fields = {**flatted_fields, **field["properties"]}
            nfields.append(flatted_fields)
        nfields_df = pd.DataFrame(nfields)
        st.write(nfields_df)
    else:
        st.write(str(summary))

    # Step 4 - Generate goals
    if summary:
        st.sidebar.write("### Goal Selection")

        num_goals = st.sidebar.slider(
            "Number of goals to generate",
            min_value=1,
            max_value=10,
            value=4)
        own_goal = st.sidebar.checkbox("Add Your Own Goal")

        # **** lida.goals *****
        goals = lida.goals(summary, n=num_goals, textgen_config=textgen_config)
        st.write(f"## Goals ({len(goals)})")

        default_goal = goals[0].question
        goal_questions = [goal.question for goal in goals]

        if own_goal:
            user_goal = st.sidebar.text_input("Describe Your Goal")

            if user_goal:

                new_goal = Goal(question=user_goal,
                                visualization=user_goal, rationale="")
                goals.append(new_goal)
                goal_questions.append(new_goal.question)

        selected_goal = st.selectbox(
            'Choose a generated goal', options=goal_questions, index=0)

        # st.markdown("### Selected Goal")
        selected_goal_index = goal_questions.index(selected_goal)
        st.write(goals[selected_goal_index])

        selected_goal_object = goals[selected_goal_index]

        # Step 5 - Generate visualizations
        if selected_goal_object:
            st.sidebar.write("## Visualization Library")
            visualization_libraries = ["seaborn", "matplotlib", "plotly"]

            selected_library = st.sidebar.selectbox(
                'Choose a visualization library',
                options=visualization_libraries,
                index=0
            )

            # Update the visualization generation call to use the selected library.
            st.write("## Visualizations")

            # slider for number of visualizations
            num_visualizations = st.sidebar.slider(
                "Number of visualizations to generate",
                min_value=1,
                max_value=10,
                value=2)

            textgen_config = TextGenerationConfig(
                n=num_visualizations, temperature=temperature,
                model=selected_model,
                use_cache=use_cache)

            # **** lida.visualize *****
            visualizations = lida.visualize(
                summary=summary,
                goal=selected_goal_object,
                textgen_config=textgen_config,
                library=selected_library)

            viz_titles = [
                f'Visualization {i+1}' for i in range(len(visualizations))]

            selected_viz_title = st.selectbox(
                'Choose a visualization', options=viz_titles, index=0)

            selected_viz = visualizations[viz_titles.index(selected_viz_title)]

            if selected_viz.raster:
                from PIL import Image
                import io
                import base64

                imgdata = base64.b64decode(selected_viz.raster)
                img = Image.open(io.BytesIO(imgdata))
                st.image(img, caption=selected_viz_title,
                         use_column_width=True)

            st.write("### Visualization Code")
            st.code(selected_viz.code)
