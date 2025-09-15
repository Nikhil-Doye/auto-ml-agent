from pyexpat import model
from unicodedata import numeric
import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
import io
import os
from openai import OpenAI
from e2b_code_interpreter import Sandbox

load_dotenv()

st.title("Automatic ML Agent")
st.markdown("Upload a csv file to get started - Only Tabular Data")

uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])

def summarize_dataset(dataframe: pd.DataFrame) -> str:
    """
    This function creates a detailed text summary that includes:
    - Column datatypes and schema information
    - Missing value counts and data completeness
    - Cardinality(unique value counts) for each column
    - Statistical summaries for numerical columns
    - Sample data rows in csv format

    Args:
        dataframe: The pandas DataFrame to summarize
    
    Returns:
        A formatted string containing the summary
    """
    try:
        buffer = io.StringIO()
        sample_rows = min(30, len(dataframe))

        dataframe.head(sample_rows).to_csv(buffer, index=False)
        sample_csv = buffer.getvalue()

        dtypes = dataframe.dtypes.astype(str).to_dict()

        non_null_counts = dataframe.notnull().sum().to_dict()

        null_counts = dataframe.isnull().sum().to_dict()

        nunique = dataframe.nunique(dropna=True).to_dict()

        numeric_cols = [col for col in dataframe.columns if pd.api.types.is_numeric_dtype(dataframe[col])]

        desc = dataframe[numeric_cols].describe().to_dict() if numeric_cols else None

        lines = []

        lines.append("Schema (dtypes):")
        for k, v in dtypes.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

        lines.append("Null/Non-Null counts:")
        for c in dataframe.columns:
            lines.append(f"- {c}: nulls={null_counts[c]}, non-nulls={int(non_null_counts[c])}")
        lines.append("")

        lines.append("Cardinality (unique values):")
        for k,v in nunique.items():
            lines.append(f"- {k}: {int(v)}")
        lines.append("")

        if desc:
            lines.append("Numerical summary stats (describe):")
            for col, stats in desc.items():
                stat_line = ", ".join([f"{s}:{round(float(val), 4) if pd.notnull(val) else 'nan'}" for s, val in stats.items()])
                lines.append(f"- {col}: {stat_line}")
        lines.append("")

        lines.append("Sample Rows (CSV Head):")
        lines.append(sample_csv)

        return "\n".join(lines)
    
    except Exception as e:
        return f"Error summarizing dataset: {e}"

def determine_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Determine if the task is classification or regression based on target column
    
    Args:
        df: The dataframe
        target_col: The target column name
    
    Returns:
        'classification' or 'regression'
    """
    if pd.api.types.is_numeric_dtype(df[target_col]):
        unique_vals = df[target_col].nunique()
        # If less than 10 unique values and all integers, likely classification
        if unique_vals <= 10 and df[target_col].dtype in ['int64', 'int32']:
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'

def build_cleaning_prompt(df, selected_col):
    dataset_summary = summarize_dataset(df.drop(columns=[selected_col]))
    prompt = f"""
    You are an expert data scientist, specifically in the field of data cleaning.
    You are given a dataframe and you need to clean the data.

    the data summary is as follows:
    {dataset_summary}

    Please clean the data and return the cleaned dataframe.
    Make sure to handle the following:
    - Missing values
    - Duplicate values
    - Outliers
    - Categorical values
    - Standardisation/Normalization

    Generate a standalone python script to clean the data, based on the data summary provided in a json property called "script".
    
    Make sure to load the data from the csv file called "input.csv" on the path "tmp/input.csv"
    The script should be a python script that can be executed to clean the data.
    Make sure to save the cleaned data to a new csv file called "cleaned.csv" on the path "tmp/cleaned.csv".
    """
    return prompt

def build_model_training_prompt(df_summary: str, target_col: str, task_type: str) -> str:
    """
    Build prompt for model training code generation with explicit file path requirements
    """
    
    models_suggestion = {
        'classification': [
            'RandomForestClassifier',
            'LogisticRegression', 
            'XGBClassifier',
            'SVC',
            'GradientBoostingClassifier'
        ],
        'regression': [
            'RandomForestRegressor',
            'LinearRegression',
            'XGBRegressor',
            'SVR', 
            'GradientBoostingRegressor'
        ]
    }
    
    metrics_suggestion = {
        'classification': ['accuracy_score', 'precision_score', 'recall_score', 'f1_score', 'roc_auc_score'],
        'regression': ['mean_squared_error', 'mean_absolute_error', 'r2_score']
    }
    
    threshold_suggestion = {
        'classification': 'accuracy > 0.75',
        'regression': 'r2_score > 0.6'
    }
    
    prompt = f"""
    You are a senior ML engineer. Generate a SINGLE Python script that trains/evaluates multiple models on tabular data.

    CONTEXT
    - Dataset summary: {df_summary}
    - Target column: {target_col}
    - Task type: {task_type}
    - Split: 80% train, 5% val, 15% test (use stratify for classification)

    HARD FILE RULES (MUST follow exactly)
    - Read input from: tmp/cleaned.csv
    - Save results to: tmp/results.json
    - Save model files to: tmp/model_{{ModelName}}.pkl
    - Always create: os.makedirs("tmp", exist_ok=True)

    SCRIPT MUST START EXACTLY WITH:
    import pandas as pd
    import numpy as np
    import json
    import joblib
    import os
    import traceback
    from sklearn.model_selection import train_test_split

    [add other needed imports, e.g., imbalanced-learn]
    results = {{
    "overall_status": "started",
    "successful_models": {{}},
    "failed_models": {{}},
    "task_type": "{task_type}",
    "target_column": "{target_col}",
    "errors": []
    }}
    os.makedirs("tmp", exist_ok=True)
    try:
    df = pd.read_csv("tmp/cleaned.csv")

    sql
    Copy code

    SCRIPT MUST END EXACTLY WITH:
    bash
    Copy code
    results["overall_status"] = "completed"
    print("=== SCRIPT VERIFICATION ===")
    print(f"Results file exists: {{os.path.exists('tmp/results.json')}}")
    print(f"Successful models: {{len(results['successful_models'])}}")
    print(f"Failed models: {{len(results['failed_models'])}}")
    except Exception as e:
    results["overall_status"] = "failed"
    results["errors"].append(f"Script error: {{str(e)}}")
    print(f"Script failed with error: {{str(e)}}")
    finally:
    with open("tmp/results.json", "w") as f:
    json.dump(results, f, indent=4)
    print("Training completed. Results saved to tmp/results.json")

    python
    Copy code

    TRAINING REQUIREMENTS
    - Models to try: {", ".join(models_suggestion[task_type])}
    - Metrics to report: {", ".join(metrics_suggestion[task_type])}
    - Threshold for success: {threshold_suggestion[task_type]}
    - Preprocessing: scale numeric, encode categorical; keep column order stable for inference
    - Split into train/val/test exactly as specified (stratify for classification)

    IMBALANCE (classification only)
    - Print df['{target_col}'].value_counts(). If max/min > 3:
    - Prefer SMOTE; else use class_weight='balanced' where supported
    - Show class distribution before/after

    EVALUATION & SAVING
    - Train each model in its own try/except; failures go to results["failed_models"]
    - Compute and store all listed metrics as floats
    - Determine meets_threshold using the threshold above
    - ONLY save models that meet the threshold to tmp/model_{{ModelName}}.pkl
    - For each model, record:
    - metrics, "meets_threshold": true/false, "model_saved": true/false
    - any error messages for failures

    VALIDATION CHECKS (must implement)
    - Use pd.read_csv("tmp/cleaned.csv")
    - Always write results with open("tmp/results.json","w") in finally
    - List saved model files found in tmp/ (print names)
    - No other input/output paths are allowed
    - Return the script as a JSON object with key "script"
    """ 
    
    return prompt

def get_deepseek_script(prompt: str) -> str:
    """
    Get generated script from DeepSeek model via OpenRouter
    """
    try:
        client = OpenAI(base_url="https://api.deepseek.com", 
                        api_key=os.getenv("OPENAI_API_KEY"))

        resp = client.chat.completions.create(
            model = "deepseek-chat",
            messages = [ 
                {"role": "system",
                "content": ( "You are a senior data scientist. Always return a strict JSON object matching the user's requested schema.")},
                {"role": "user", "content": prompt}
            ],
            response_format = {"type": "json_object"}
        )
        if not resp or not getattr(resp, 'choices', None):
            return None

        text = resp.choices[0].message.content or ""

        try:
            data = json.loads(text)
            script_val = data.get("script")
            if isinstance(script_val, str) and script_val.strip():
                return script_val.strip()
        except Exception as e:
            print(f"Error parsing script: {text} {e}")

    except Exception as e:
        print(f"Error getting script: {e}")
        return None

def execute_cleaning_in_e2b(script: str, csv_bytes: bytes):
    """
    Execute data cleaning script in E2B sandbox
    
    Args:
        script: Python cleaning script
        csv_bytes: Raw CSV data as bytes
    
    Returns:
        Tuple of (cleaned_csv_bytes, execution_info)
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        raise ValueError("E2B_API_KEY is not set")
    
    sandbox = Sandbox.create()
    exec_info = {
        'operation': 'data_cleaning',
        'status': 'started'
    }

    try:
        # Ensure tmp directory exists
        sandbox.run_code("import os; os.makedirs('tmp', exist_ok=True)")
        
        # Write input CSV data
        sandbox.files.write("tmp/input.csv", csv_bytes)
        exec_info['input_file_written'] = True
        
        # Execute cleaning script
        result = sandbox.run_code(script)
        exec_info['exit_code'] = getattr(result, "exit_code", 0)
        exec_info['stdout'] = getattr(result, "stdout", "")
        exec_info['stderr'] = getattr(result, "stderr", "")
        
        # Check if cleaning was successful
        if exec_info['exit_code'] == 0:
            try:
                # Read cleaned CSV
                cleaned_bytes = sandbox.files.read('tmp/cleaned.csv')
                exec_info['status'] = 'success'
                exec_info['cleaned_file_size'] = len(cleaned_bytes)
                return cleaned_bytes, exec_info
                
            except Exception as e:
                exec_info['status'] = 'failed'
                exec_info['error'] = f"Could not read cleaned.csv: {str(e)}"
                return None, exec_info
        else:
            exec_info['status'] = 'failed'
            exec_info['error'] = f"Script execution failed with exit code {exec_info['exit_code']}"
            return None, exec_info
    
    except Exception as e:
        exec_info['status'] = 'error'
        exec_info['error'] = f"Sandbox execution error: {str(e)}"
        return None, exec_info
    
    # finally:
    #     try:
    #         sandbox.close()
    #     except Exception as e:
    #         exec_info['sandbox_close_error'] = str(e)

def execute_training_in_e2b(script: str, cleaned_csv_bytes: bytes):
    """
    Execute model training script in E2B sandbox
    
    Args:
        script: Python training script
        cleaned_csv_bytes: Cleaned CSV data as bytes
    
    Returns:
        Tuple of (training_results, execution_info)
    """
    api_key = os.getenv("E2B_API_KEY")
    if not api_key:
        raise ValueError("E2B_API_KEY is not set")
    
    sandbox = Sandbox.create()
    exec_info = {
        'operation': 'model_training',
        'status': 'started'
    }

    try:
        # Ensure tmp directory exists
        sandbox.run_code("import os; os.makedirs('tmp', exist_ok=True)")
        
        # Write cleaned CSV data
        sandbox.files.write("tmp/cleaned.csv", cleaned_csv_bytes)
        exec_info['cleaned_file_written'] = True
        
        # Execute training script with timeout
        try:
            result = sandbox.run_code(script, timeout=300)  # 5 minute timeout
            exec_info['exit_code'] = getattr(result, "exit_code", 0)
            exec_info['stdout'] = getattr(result, "stdout", "")
            exec_info['stderr'] = getattr(result, "stderr", "")
            
            # Log the full output for debugging
            if exec_info['stdout']:
                print("STDOUT:", exec_info['stdout'])
            if exec_info['stderr']:
                print("STDERR:", exec_info['stderr'])
                
        except Exception as e:
            exec_info['script_execution_error'] = str(e)
            exec_info['exit_code'] = -1
        
        # Initialize training results with default structure
        training_results = {
            'results': {
                'successful_models': {},
                'failed_models': {},
                'overall_status': 'unknown',
                'errors': []
            },
            'models': {}
        }
        
        # Always try to read results, even if script had non-zero exit code
        try:
            # Enhanced file listing for debugging
            files_result = sandbox.run_code("""
            import os
            try:
                if os.path.exists('tmp'):
                    files = os.listdir('tmp')
                    print(f'Files in tmp: {files}')
                    for f in files:
                        full_path = os.path.join('tmp', f)
                        if os.path.isfile(full_path):
                            size = os.path.getsize(full_path)
                            print(f'  {f}: {size} bytes')
                else:
                    print('tmp directory does not exist')
            except Exception as e:
                print(f'Error listing files: {e}')
            """)
            print(files_result)
            exec_info['tmp_files'] = getattr(files_result, 'stdout', 'Could not list files')
            
            # Try to read results.json
            results_bytes = sandbox.files.read('tmp/results.json')
            results_data = json.loads(results_bytes)#.decode('utf-8')
            training_results['results'] = results_data
            exec_info['results_read'] = True
            
        except FileNotFoundError:
            exec_info['results_read'] = False
            exec_info['results_error'] = "results.json not found"
            training_results['results']['overall_status'] = 'failed'
            training_results['results']['errors'].append("Script did not generate results.json")
            
        except json.JSONDecodeError as e:
            exec_info['results_read'] = False
            exec_info['results_error'] = f"Invalid JSON in results.json: {str(e)}"
            
        except Exception as e:
            exec_info['results_read'] = False
            exec_info['results_error'] = f"Error reading results.json: {str(e)}"
        
        # Try to download saved models
        model_files_to_check = [
            'RandomForestClassifier', 'LogisticRegression', 'XGBClassifier', 
            'SVC', 'GradientBoostingClassifier', 'RandomForestRegressor',
            'LinearRegression', 'XGBRegressor', 'SVR', 'GradientBoostingRegressor'
        ]
        
        models_downloaded = 0
        for model_name in model_files_to_check:
            try:
                model_bytes = sandbox.files.read(f'tmp/model_{model_name}.pkl')
                training_results['models'][model_name] = model_bytes
                models_downloaded += 1
                
            except FileNotFoundError:
                continue  # Model wasn't saved (likely didn't meet threshold)
                
            except Exception as e:
                exec_info[f'model_download_error_{model_name}'] = str(e)
        
        exec_info['models_downloaded'] = models_downloaded
        
        # Determine overall status
        if exec_info['exit_code'] == 0 and exec_info.get('results_read', False):
            exec_info['status'] = 'success'
        elif models_downloaded > 0 or exec_info.get('results_read', False):
            exec_info['status'] = 'partial_success'
        else:
            exec_info['status'] = 'failed'
            
        return training_results, exec_info
    
    except Exception as e:
        exec_info['status'] = 'error'
        exec_info['error'] = f"Sandbox execution error: {str(e)}"
        return training_results, exec_info
    
    # finally:
    #     try:
    #         sandbox.close()
    #     except Exception as e:
    #         exec_info['sandbox_close_error'] = str(e)

def validate_training_script(script: str) -> bool:
    """
    Validate that the training script contains essential components
    """
    if not script or len(script.strip()) < 100:
        return False
    
    required_elements = [
        'pd.read_csv("tmp/cleaned.csv")',
        'tmp/results.json',
        'os.makedirs',
        'json.dump',
        'finally:'
    ]
    
    for element in required_elements:
        if element not in script:
            st.error(f"Generated script missing required element: {element}")
            return False
    
    return True

def run_cleaning_workflow(df: pd.DataFrame, selected_col: str):
    """
    Execute the data cleaning workflow
    
    Args:
        df: Input dataframe
        selected_col: Target column name
        
    Returns:
        Tuple of (cleaned_df, execution_info, success)
    """
    try:
        # Generate cleaning prompt
        cleaning_prompt = build_cleaning_prompt(df, selected_col)
        
        # Get cleaning script from LLM
        cleaning_script = get_deepseek_script(cleaning_prompt)
        
        if not cleaning_script:
            return None, {'error': 'Failed to generate cleaning script'}, False
        
        # Convert dataframe to CSV bytes
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        
        # Execute cleaning
        cleaned_bytes, exec_info = execute_cleaning_in_e2b(cleaning_script, csv_bytes)
        
        if cleaned_bytes and exec_info['status'] == 'success':
            cleaned_df = pd.read_csv(io.StringIO(cleaned_bytes))#.decode('utf-8')
            return cleaned_df, exec_info, True
        else:
            return None, exec_info, False
            
    except Exception as e:
        return None, {'error': f'Cleaning workflow error: {str(e)}'}, False

def run_training_workflow(cleaned_df: pd.DataFrame, selected_col: str, task_type: str):
    """
    Execute the model training workflow
    
    Args:
        cleaned_df: Cleaned dataframe
        selected_col: Target column name
        task_type: 'classification' or 'regression'
        
    Returns:
        Tuple of (training_results, execution_info, success)
    """
    try:
        # Generate training summary and prompt
        training_summary = summarize_dataset(cleaned_df)
        training_prompt = build_model_training_prompt(training_summary, selected_col, task_type)
        
        # Get training script from LLM
        training_script = get_deepseek_script(training_prompt)
        
        if not training_script:
            return None, {'error': 'Failed to generate training script'}, False
        
        # Validate training script
        if not validate_training_script(training_script):
            st.error("Generated script failed validation. Please try again.")
            with st.expander("Generated Script (Invalid)"):
                st.code(training_script, language='python')
            return None, {'error': 'Generated script failed validation'}, False
        
        # Convert cleaned dataframe to CSV bytes
        cleaned_csv_bytes = cleaned_df.to_csv(index=False).encode('utf-8')
        
        # Execute training
        training_results, exec_info = execute_training_in_e2b(training_script, cleaned_csv_bytes)
        
        # Determine success based on status
        success = exec_info['status'] in ['success', 'partial_success']
        
        return training_results, exec_info, success
        
    except Exception as e:
        return None, {'error': f'Training workflow error: {str(e)}'}, False

def display_workflow_results(step_name: str, exec_info: dict, success: bool):
    """
    Display workflow step results in Streamlit
    
    Args:
        step_name: Name of the workflow step
        exec_info: Execution information dictionary
        success: Whether the step was successful
    """
    if success:
        st.success(f"✅ {step_name} completed successfully!")
    else:
        st.error(f"❌ {step_name} failed!")
    
    # Show execution details
    with st.expander(f"{step_name} Execution Details"):
        st.json(exec_info)
    
    # Show stdout if available
    if exec_info.get('stdout'):
        with st.expander(f"{step_name} Output"):
            st.text(exec_info['stdout'])
    
    # Show stderr if available  
    if exec_info.get('stderr'):
        with st.expander(f"{step_name} Errors"):
            st.text(exec_info['stderr'])

def display_model_results(training_results: dict):
    """
    Display model training results in Streamlit - WITHOUT download buttons
    """
    if not training_results or 'results' not in training_results:
        st.error("No training results available")
        return
    
    results = training_results['results']
    models_data = training_results.get('models', {})
    
    st.subheader("Model Training Results")
    
    # Show all models in a comprehensive table
    all_models = {}
    if 'successful_models' in results:
        all_models.update(results['successful_models'])
    if 'failed_models' in results:
        all_models.update(results['failed_models'])
    
    if all_models:
        st.write("### All Trained Models")
        model_metrics = []
        for model_name, metrics in all_models.items():
            model_row = {'Model': model_name}
            model_row.update(metrics)
            # Add status indicator
            model_row['Status'] = '✅ Above Threshold' if metrics.get('meets_threshold', False) else '❌ Below Threshold'
            model_metrics.append(model_row)
        
        if model_metrics:
            metrics_df = pd.DataFrame(model_metrics)
            # Sort by performance (assuming accuracy or r2_score)
            if 'accuracy' in metrics_df.columns:
                metrics_df = metrics_df.sort_values('accuracy', ascending=False)
            elif 'r2_score' in metrics_df.columns:
                metrics_df = metrics_df.sort_values('r2_score', ascending=False)
            
            st.dataframe(metrics_df, use_container_width=True)
    
    # Show successful models section WITHOUT download buttons
    if 'successful_models' in results and results['successful_models']:
        st.write("### 🏆 Recommended Models (Above Threshold)")
        st.success(f"**{len(results['successful_models'])}** models met the performance threshold and are recommended for use!")
        
        # Show model details without download buttons
        for model_name, metrics in results['successful_models'].items():
            with st.container():
                st.write(f"#### {model_name}")
                
                # Display metrics in columns
                if 'accuracy' in metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    with col2:
                        if 'precision' in metrics:
                            st.metric("Precision", f"{metrics['precision']:.3f}")
                    with col3:
                        if 'recall' in metrics:
                            st.metric("Recall", f"{metrics['recall']:.3f}")
                    with col4:
                        if 'f1_score' in metrics:
                            st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                
                elif 'r2_score' in metrics:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² Score", f"{metrics['r2_score']:.3f}")
                    with col2:
                        if 'mean_squared_error' in metrics:
                            st.metric("MSE", f"{metrics['mean_squared_error']:.3f}")
                    with col3:
                        if 'mean_absolute_error' in metrics:
                            st.metric("MAE", f"{metrics['mean_absolute_error']:.3f}")
                
                st.write("---")
    else:
        st.warning("⚠️ No models met the performance threshold requirements")
        st.info("💡 All models have been trained and saved internally for API service use.")
    
    # Show failed models summary (optional)
    if 'failed_models' in results and results['failed_models']:
        with st.expander("📊 View Models Below Threshold"):
            st.warning("⚠️ These models performed below the recommended threshold:")
            
            failed_metrics = []
            for model_name, metrics in results['failed_models'].items():
                model_row = {'Model': model_name}
                model_row.update(metrics)
                failed_metrics.append(model_row)
            
            if failed_metrics:
                failed_df = pd.DataFrame(failed_metrics)
                st.dataframe(failed_df, use_container_width=True)
    
    # Show model count summary
    total_models = len(all_models)
    successful_count = len(results.get('successful_models', {}))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Models Trained", total_models)
    with col2:
        st.metric("Models Above Threshold", successful_count)
    with col3:
        success_rate = (successful_count / total_models * 100) if total_models > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    # Info message about model availability
    st.info("🔒 **Models are securely stored and available through the API service for predictions.**")
        
# Main Streamlit App
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    selected_col = st.selectbox("Select a column to predict", 
                    df.columns.tolist(), 
                    help="The column you want to predict")

    if selected_col:
        task_type = determine_task_type(df, selected_col)
        st.info(f"Detected task type: **{task_type.title()}**")

    # Workflow-based approach with separate buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        clean_only_button = st.button("Clean Data Only", type="secondary")
    
    with col2:
        # Check session state in real-time
        is_cleaned_data_available = st.session_state.get('cleaned_data_available', False)
        train_only_button = st.button("Train Models Only", type="secondary", 
                                    disabled=not is_cleaned_data_available)
    
    with col3:
        full_pipeline_button = st.button("Full Pipeline", type="primary")

    # Add helper text below buttons
    if not st.session_state.get('cleaned_data_available', False):
        st.info("Clean your data first to enable model training")
    else:
        st.success("Cleaned data available - you can now train models!")

    # Workflow 1: Data Cleaning Only
    if clean_only_button and selected_col:
        st.write("## Data Cleaning Workflow")
        
        with st.spinner("Generating cleaning script..."):
            # Show cleaning prompt
            cleaning_prompt = build_cleaning_prompt(df, selected_col)
            with st.expander("Cleaning Prompt"):
                st.write(cleaning_prompt)
            
            # Get and show cleaning script
            cleaning_script = get_deepseek_script(cleaning_prompt)
            with st.expander("Generated Cleaning Script"):
                st.code(cleaning_script, language='python')
        
        with st.spinner("Executing data cleaning workflow..."):
            cleaned_df, exec_info, success = run_cleaning_workflow(df, selected_col)
            
            # Display workflow results
            display_workflow_results("Data Cleaning", exec_info, success)
            
            if success and cleaned_df is not None:
                # Store in session state for potential training FIRST
                st.session_state['cleaned_df'] = cleaned_df
                st.session_state['cleaned_data_available'] = True
                st.session_state['selected_col'] = selected_col
                st.session_state['task_type'] = task_type
                
                # Show cleaned data
                with st.expander("Cleaned Dataset Preview"):
                    st.dataframe(cleaned_df.head())
                    st.write(f"**Shape:** {cleaned_df.shape}")
                
                st.success("Data cleaning completed! You can now train models using the 'Train Models Only' button.")
                
                # Force a rerun to update the button state
                st.rerun()

    # Workflow 2: Training Only (using previously cleaned data)
    if train_only_button and st.session_state.get('cleaned_data_available', False):
        st.write("## Model Training Workflow")
        
        # Get data from session state
        cleaned_df = st.session_state['cleaned_df']
        selected_col = st.session_state['selected_col']
        task_type = st.session_state['task_type']
        
        st.info(f"Using previously cleaned data. Target: **{selected_col}**, Task: **{task_type.title()}**")
        
        with st.spinner("Generating training script..."):
            # Show training prompt
            training_summary = summarize_dataset(cleaned_df)
            training_prompt = build_model_training_prompt(training_summary, selected_col, task_type)
            with st.expander("Training Prompt"):
                st.write(training_prompt)
            
            # Get and show training script
            training_script = get_deepseek_script(training_prompt)
            with st.expander("Generated Training Script"):
                st.code(training_script, language='python')
                
                # Add debug info
                if training_script:
                    st.write(f"**Script Length:** {len(training_script)} characters")
                    st.write(f"**Contains results.json:** {'tmp/results.json' in training_script}")
                    st.write(f"**Contains model saving:** {'joblib.dump' in training_script}")
                else:
                    st.error("No script was generated!")
        
        with st.spinner("Executing model training workflow..."):
            training_results, exec_info, success = run_training_workflow(cleaned_df, selected_col, task_type)
            
            # Display workflow results
            display_workflow_results("Model Training", exec_info, success)
            
            if success and training_results:
                display_model_results(training_results)

    # Workflow 3: Full Pipeline
    if full_pipeline_button and selected_col:
        st.write("## Full Auto ML Pipeline")
        
        # Step 1: Data Cleaning
        st.write("### Step 1: Data Cleaning")
        with st.spinner("Executing cleaning workflow..."):
            cleaned_df, clean_exec_info, clean_success = run_cleaning_workflow(df, selected_col)
            
            display_workflow_results("Data Cleaning", clean_exec_info, clean_success)
            
            if not clean_success or cleaned_df is None:
                st.error("Cannot proceed to model training due to cleaning failure.")
                st.stop()
            
            st.success("Data cleaning completed successfully!")
            
            # Show cleaned data preview
            with st.expander("Cleaned Dataset Preview"):
                st.dataframe(cleaned_df.head())
                st.write(f"**Shape:** {cleaned_df.shape}")
        
        # Step 2: Model Training
        st.write("### Step 2: Model Training")
        with st.spinner("Executing training workflow..."):
            training_summary = summarize_dataset(cleaned_df)
            training_prompt = build_model_training_prompt(training_summary, selected_col, task_type)
            with st.expander("Training Prompt"):
                st.write(training_prompt)
            
            training_script = get_deepseek_script(training_prompt)
            with st.expander("Generated Training Script"):
                st.code(training_script, language='python')
            
            training_results, train_exec_info, train_success = run_training_workflow(cleaned_df, selected_col, task_type)
            
            display_workflow_results("Model Training", train_exec_info, train_success)
            
            if train_success and training_results:
                display_model_results(training_results)
            
        # Pipeline completion summary
        st.write("### Pipeline Summary")
        
        if clean_success and train_success:
            st.success("🎉 Full Auto ML Pipeline completed successfully!")
            
            # Store final results in session state
            st.session_state['cleaned_df'] = cleaned_df
            st.session_state['cleaned_data_available'] = True
            st.session_state['selected_col'] = selected_col
            st.session_state['task_type'] = task_type
            
        elif clean_success and not train_success:
            st.warning("Data cleaning succeeded, but model training failed.")
            st.info("Your data has been cleaned and is available for manual training.")
            
        else:
            st.error("Pipeline failed at the data cleaning stage.")