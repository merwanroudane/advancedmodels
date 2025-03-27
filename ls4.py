import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.regression.rolling import RollingOLS
from statsmodels.graphics.regressionplots import plot_leverage_resid2
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import variance_inflation_factor
import io
import base64
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib
import itertools

matplotlib.use('Agg')

# For dominance analysis
def dominance_analysis(X, y, model_type="ols"):
    """
    Perform dominance analysis to determine variable importance
    """
    variables = X.columns.tolist()
    n_vars = len(variables)

    all_combinations = []
    for i in range(n_vars + 1):
        all_combinations.extend(list(itertools.combinations(variables, i)))

    all_combinations = all_combinations[1:]

    r2_dict = {}
    for combo in all_combinations:
        if len(combo) == 0:
            continue
        X_subset = X[list(combo)]
        X_with_const = sm.add_constant(X_subset)
        if model_type == "ols":
            model = sm.OLS(y, X_with_const)
            results = model.fit()
            r2_dict[combo] = results.rsquared

    total_contributions = {var: 0 for var in variables}
    counts = {var: 0 for var in variables}

    for var in variables:
        for combo in all_combinations:
            if var not in combo:
                continue
            combo_without_var = tuple(x for x in combo if x != var)
            if combo_without_var:
                contribution = r2_dict[combo] - r2_dict[combo_without_var]
            else:
                contribution = r2_dict[combo]
            total_contributions[var] += contribution
            counts[var] += 1

    avg_contributions = {var: total_contributions[var] / counts[var] for var in variables}
    dominance_stats = pd.DataFrame({
        'Variable': list(avg_contributions.keys()),
        'Average_Contribution': list(avg_contributions.values())
    })
    dominance_stats = dominance_stats.sort_values('Average_Contribution', ascending=False)
    return dominance_stats

# Function to create formatted regression summary
def format_regression_results(results):
    """Format regression results in a nice HTML table"""
    r2 = results.rsquared
    adj_r2 = results.rsquared_adj
    f_stat = results.fvalue
    f_pvalue = results.f_pvalue
    nobs = results.nobs

    coef_df = pd.DataFrame({
        'Coefficient': results.params,
        'Std Error': results.bse,
        't-value': results.tvalues,
        'p-value': results.pvalues,
        '[0.025': results.conf_int()[0],
        '0.975]': results.conf_int()[1]
    })

    def format_pvalue(pvalue):
        if pvalue < 0.001:
            return f"{pvalue:.4f} ***"
        elif pvalue < 0.01:
            return f"{pvalue:.4f} **"
        elif pvalue < 0.05:
            return f"{pvalue:.4f} *"
        elif pvalue < 0.1:
            return f"{pvalue:.4f} ."
        else:
            return f"{pvalue:.4f}"

    coef_df['p-value'] = coef_df['p-value'].apply(format_pvalue)

    html = f"""
    <h3>Regression Results</h3>
    <div style="background-color:#f5f5f5; padding:15px; border-radius:5px; margin-bottom:15px">
        <table style="width:100%">
            <tr>
                <td><strong>R-squared:</strong></td>
                <td>{r2:.4f}</td>
                <td><strong>Adjusted R-squared:</strong></td>
                <td>{adj_r2:.4f}</td>
            </tr>
            <tr>
                <td><strong>F-statistic:</strong></td>
                <td>{f_stat:.4f}</td>
                <td><strong>Prob (F-statistic):</strong></td>
                <td>{f_pvalue:.4g}</td>
            </tr>
            <tr>
                <td><strong>Number of Observations:</strong></td>
                <td>{int(nobs)}</td>
                <td><strong>AIC:</strong></td>
                <td>{results.aic:.4f}</td>
            </tr>
            <tr>
                <td><strong>BIC:</strong></td>
                <td>{results.bic:.4f}</td>
                <td><strong>Log-Likelihood:</strong></td>
                <td>{results.llf:.4f}</td>
            </tr>
        </table>
    </div>
    """
    return html, coef_df

# Set page title and layout
st.set_page_config(
    page_title="Regression Analysis Tool",
    layout="wide"
)

# App title and description
st.title("Comprehensive Regression Analysis Tool")
st.markdown("""
This application allows you to perform various regression analyses on your data:
- Data exploration with correlation heatmap
- Linear regression with diagnostic tests
- Rolling regression analysis
- Recursive least squares
- Dominance analysis for variable importance
""")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        st.sidebar.subheader("Data Information")
        st.sidebar.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        if not isinstance(df.index, pd.DatetimeIndex):
            date_columns = df.select_dtypes(include=['datetime64']).columns.tolist()
            if len(date_columns) > 0:
                date_column = st.sidebar.selectbox("Select date column for index", date_columns)
                if date_column:
                    df = df.set_index(date_column)
                    df.index = pd.to_datetime(df.index)
            else:
                st.sidebar.warning("No datetime columns found. Some time series analyses might not work properly.")

        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        if len(numeric_columns) > 0:
            dependent_var = st.sidebar.selectbox("Select Dependent Variable (Y)", numeric_columns)
            remaining_columns = [col for col in numeric_columns if col != dependent_var]
            independent_vars = st.sidebar.multiselect("Select Independent Variables (X)", remaining_columns,
                                                    default=remaining_columns[:min(3, len(remaining_columns))])

            if dependent_var and len(independent_vars) > 0:
                X = df[independent_vars]
                y = df[dependent_var]
                data = pd.concat([y, X], axis=1).dropna()
                if data.empty:
                    st.error("No valid data after removing NaN values. Please check your dataset.")
                else:
                    X = data[independent_vars]
                    y = data[dependent_var]

                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Data Exploration",
                        "Linear Regression & Diagnostics",
                        "Rolling Regression",
                        "Recursive Least Squares",
                        "Dominance Analysis"
                    ])

                    # Tab 1: Data Exploration
                    with tab1:
                        st.header("Data Exploration")
                        st.subheader("Preview of the Dataset")
                        st.dataframe(df.head())
                        st.subheader("Summary Statistics")
                        st.dataframe(df[numeric_columns].describe())
                        st.subheader("Correlation Heatmap")
                        selected_cols = independent_vars + [dependent_var]
                        corr = df[selected_cols].corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                        st.pyplot(fig)
                        st.subheader("Pairplot of Selected Variables")
                        if st.button("Generate Pairplot (may take time for large datasets)"):
                            with st.spinner("Generating pairplot..."):
                                fig = sns.pairplot(df[selected_cols], diag_kind="kde")
                                st.pyplot(fig)

                    # Tab 2: Linear Regression & Diagnostics
                    with tab2:
                        st.header("Linear Regression Analysis")
                        X_with_const = sm.add_constant(X)
                        model = sm.OLS(y, X_with_const)
                        results = model.fit()
                        summary_html, coef_df = format_regression_results(results)
                        st.markdown(summary_html, unsafe_allow_html=True)
                        st.subheader("Coefficient Table")
                        st.dataframe(coef_df.style.format({
                            'Coefficient': '{:.4f}',
                            'Std Error': '{:.4f}',
                            't-value': '{:.4f}',
                            '[0.025': '{:.4f}',
                            '0.975]': '{:.4f}'
                        }))
                        st.markdown("""
                        **Significance levels:** 
                        - *** : p < 0.001
                        - ** : p < 0.01
                        - * : p < 0.05
                        - . : p < 0.1
                        """)
                        with st.expander("View detailed statsmodels summary"):
                            st.text(str(results.summary()))
                        st.subheader("Regression Diagnostics")
                        st.write("**Normality of Residuals**")
                        omni_test = sms.omni_normtest(results.resid)
                        st.write(f"Omni Test: Chi² = {omni_test[0]:.4f}, p-value = {omni_test[1]:.4f}")
                        jb_test = sms.jarque_bera(results.resid)
                        st.write(f"Jarque-Bera Test: JB = {jb_test[0]:.4f}, p-value = {jb_test[1]:.4f}, Skew = {jb_test[2]:.4f}, Kurtosis = {jb_test[3]:.4f}")
                        st.write("**Residual Plots**")
                        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                        sm.qqplot(results.resid, line='45', ax=axes[0, 0])
                        axes[0, 0].set_title('QQ Plot')
                        axes[0, 1].hist(results.resid, bins=20)
                        axes[0, 1].set_title('Residual Histogram')
                        axes[1, 0].scatter(results.fittedvalues, results.resid)
                        axes[1, 0].axhline(y=0, color='r', linestyle='-')
                        axes[1, 0].set_title('Residuals vs Fitted')
                        axes[1, 0].set_xlabel('Fitted values')
                        axes[1, 0].set_ylabel('Residuals')
                        axes[1, 1].scatter(results.fittedvalues, np.sqrt(np.abs(results.resid)))
                        axes[1, 1].set_title('Scale-Location Plot')
                        axes[1, 1].set_xlabel('Fitted values')
                        axes[1, 1].set_ylabel('√|Residuals|')
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.write("**Autocorrelation**")
                        dw_stat = sms.durbin_watson(results.resid)
                        st.write(f"Durbin-Watson Statistic: {dw_stat:.4f}")
                        bg_test = sms.acorr_breusch_godfrey(results)
                        st.write(f"Breusch-Godfrey Test: LM = {bg_test[0]:.4f}, p-value = {bg_test[1]:.4f}, F = {bg_test[2]:.4f}, F p-value = {bg_test[3]:.4f}")
                        st.write("**ACF and PACF Plots**")
                        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                        plot_acf(results.resid, ax=axes[0])
                        plot_pacf(results.resid, ax=axes[1])
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.write("**Multicollinearity**")
                        vif_data = pd.DataFrame()
                        vif_data["Variable"] = X.columns
                        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                        st.dataframe(vif_data)
                        condition_number = np.linalg.cond(X_with_const)
                        st.write(f"Condition Number: {condition_number:.4f}")
                        st.write("**Heteroskedasticity Tests**")
                        bp_test = sms.het_breuschpagan(results.resid, results.model.exog)
                        st.write(f"Breusch-Pagan Test: LM = {bp_test[0]:.4f}, p-value = {bp_test[1]:.4f}, F = {bp_test[2]:.4f}, F p-value = {bp_test[3]:.4f}")
                        gq_test = sms.het_goldfeldquandt(results.resid, results.model.exog)
                        st.write(f"Goldfeld-Quandt Test: F = {gq_test[0]:.4f}, p-value = {gq_test[1]:.4f}")
                        white_test = sms.het_white(results.resid, results.model.exog)
                        st.write(f"White Test: LM = {white_test[0]:.4f}, p-value = {white_test[1]:.4f}, F = {white_test[2]:.4f}, F p-value = {white_test[3]:.4f}")
                        st.write("**Influence Tests**")
                        influence = OLSInfluence(results)
                        st.write("Leverage Plot")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        fig = plot_leverage_resid2(results, ax=ax)
                        st.pyplot(fig)
                        st.write("Cook's Distance")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.stem(influence.cooks_distance[0])
                        ax.set_title("Cook's Distance")
                        ax.set_xlabel("Observation")
                        ax.set_ylabel("Cook's Distance")
                        st.pyplot(fig)
                        st.write("DFBETAS for Selected Variables")
                        dfbetas = influence.dfbetas
                        fig, axes = plt.subplots(len(X.columns), 1, figsize=(10, 3 * len(X.columns)))
                        if len(X.columns) == 1:
                            axes = [axes]
                        for i, col in enumerate(X.columns):
                            axes[i].stem(dfbetas[:, i + 1])
                            axes[i].set_title(f"DFBETAS for {col}")
                            axes[i].axhline(y=0, color='r', linestyle='-')
                        plt.tight_layout()
                        st.pyplot(fig)
                        st.write("**Linearity Test**")
                        try:
                            harvey_collier = sms.linear_harvey_collier(results)
                            st.write(f"Harvey-Collier Test: t = {harvey_collier[0]:.4f}, p-value = {harvey_collier[1]:.4f}")
                        except:
                            st.write("Harvey-Collier test could not be computed.")

                    # Tab 3: Rolling Regression
                    with tab3:
                        st.header("Rolling Regression Analysis")
                        window_size = st.slider("Select Window Size", min_value=5, max_value=min(100, len(df) - 5),
                                                value=min(20, len(df) - 5))
                        try:
                            if isinstance(df.index, pd.DatetimeIndex):
                                st.info("Using datetime index for rolling regression.")
                            else:
                                st.warning("Data is not sorted by date. Rolling regression will use the natural order of observations.")
                            rols = RollingOLS(y, sm.add_constant(X), window=window_size)
                            rres = rols.fit()
                            st.subheader("Rolling Coefficients")
                            rolling_params = rres.params.copy()
                            st.write("**Rolling Coefficient Plots**")
                            fig, axes = plt.subplots(len(X.columns), 1, figsize=(12, 4 * len(X.columns)))
                            if len(X.columns) == 1:
                                axes = [axes]
                            for i, var in enumerate(X.columns):
                                var_idx = i + 1
                                ax = axes[i]
                                valid_params = rolling_params.iloc[:, var_idx].dropna()
                                valid_index = valid_params.index
                                ax.plot(valid_index, valid_params, label=f"{var} Coefficient")
                                try:
                                    ci = 1.96 * np.sqrt(rres.cov_params().iloc[var_idx, var_idx])
                                    upper = valid_params + ci
                                    lower = valid_params - ci
                                    ax.fill_between(valid_index, lower, upper, alpha=0.3)
                                except:
                                    pass
                                ax.set_title(f"Rolling Coefficient: {var}")
                                ax.set_xlabel("Window End Date")
                                ax.set_ylabel("Coefficient Value")
                                ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                                ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                            st.subheader("Rolling R-squared")
                            fig, ax = plt.subplots(figsize=(12, 6))
                            if hasattr(rres, 'rsquared'):
                                valid_rsq = rres.rsquared.dropna()
                                ax.plot(valid_rsq.index, valid_rsq, label="R-squared")
                                ax.set_title("Rolling R-squared")
                                ax.set_xlabel("Window End Date")
                                ax.set_ylabel("R-squared")
                                ax.legend()
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.write("R-squared values not available.")
                            if st.checkbox("Show Expanding Regression Results"):
                                st.subheader("Expanding Regression")
                                min_nobs = st.slider("Minimum Observations for First Estimate", min_value=3,
                                                    max_value=window_size, value=5)
                                expand_rols = RollingOLS(y, sm.add_constant(X), window=window_size, min_nobs=min_nobs,
                                                        expanding=True)
                                expand_rres = expand_rols.fit()
                                st.write("**Expanding Coefficient Plots**")
                                fig, axes = plt.subplots(len(X.columns), 1, figsize=(12, 4 * len(X.columns)))
                                if len(X.columns) == 1:
                                    axes = [axes]
                                for i, var in enumerate(X.columns):
                                    var_idx = i + 1
                                    ax = axes[i]
                                    exp_params = expand_rres.params.iloc[:, var_idx].dropna()
                                    exp_index = exp_params.index
                                    ax.plot(exp_index, exp_params, label=f"{var} Coefficient")
                                    ax.set_title(f"Expanding Coefficient: {var}")
                                    ax.set_xlabel("End Date")
                                    ax.set_ylabel("Coefficient Value")
                                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                                    ax.legend()
                                plt.tight_layout()
                                st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error in rolling regression analysis: {str(e)}")

                    # Tab 4: Recursive Least Squares (Corrected Version)
                    with tab4:
                        st.header("Recursive Least Squares Analysis")

                        try:
                            if len(df) < 10:
                                st.warning("Not enough data points for recursive analysis. Need at least 10 observations.")
                            else:
                                X_with_const = sm.add_constant(X)
                                mod = sm.RecursiveLS(y, X_with_const)
                                with st.spinner("Running recursive least squares analysis..."):
                                    res = mod.fit()

                                st.subheader("Model Summary")
                                summary_html = f"""
                                <div style="background-color:#f5f5f5; padding:15px; border-radius:5px;">
                                    <table style="width:100%">
                                        <tr>
                                            <td><strong>Number of Observations:</strong></td>
                                            <td>{int(res.nobs)}</td>
                                            <td><strong>Log-Likelihood:</strong></td>
                                            <td>{res.llf:.4f}</td>
                                        </tr>
                                        <tr>
                                            <td><strong>AIC:</strong></td>
                                            <td>{res.aic:.4f}</td>
                                            <td><strong>BIC:</strong></td>
                                            <td>{res.bic:.4f}</td>
                                        </tr>
                                    </table>
                                </div>
                                """
                                st.markdown(summary_html, unsafe_allow_html=True)

                                with st.expander("Detailed RLS Summary"):
                                    st.text(str(res.summary()))

                                st.subheader("Recursive Parameter Estimates")
                                fig = res.plot_recursive_coefficient(range(X_with_const.shape[1]),
                                                                   alpha=0.05,
                                                                   figsize=(10, 3 * X_with_const.shape[1]))
                                for ax in fig.axes:
                                    ax.set_xlabel("Time" if not isinstance(df.index, pd.DatetimeIndex) else "Date")
                                    ax.set_ylabel("Coefficient Value")
                                plt.tight_layout()
                                st.pyplot(fig)

                                st.subheader("CUSUM Test for Parameter Stability")
                                fig = res.plot_cusum(alpha=0.05, legend_loc='upper left')
                                if isinstance(df.index, pd.DatetimeIndex):
                                    fig.axes[0].set_xlabel("Date")
                                else:
                                    fig.axes[0].set_xlabel("Time")
                                fig.axes[0].set_ylabel("CUSUM Statistic")
                                plt.tight_layout()
                                st.pyplot(fig)
                                st.markdown("""
                                **CUSUM Test Interpretation:**
                                - If the CUSUM statistic stays within the confidence bands (red lines), 
                                  the parameters are stable.
                                - Crossing the bands suggests structural breaks or parameter instability.
                                """)

                                st.subheader("CUSUM of Squares Test")
                                fig = res.plot_cusum_squares(alpha=0.05, legend_loc='upper left')
                                if isinstance(df.index, pd.DatetimeIndex):
                                    fig.axes[0].set_xlabel("Date")
                                else:
                                    fig.axes[0].set_xlabel("Time")
                                fig.axes[0].set_ylabel("CUSUM of Squares")
                                plt.tight_layout()
                                st.pyplot(fig)
                                st.markdown("""
                                **CUSUM of Squares Interpretation:**
                                - Tests for changes in variance structure.
                                - Staying within bands indicates stable variance.
                                - Crossing bands may indicate heteroskedasticity or model misspecification.
                                """)

                                st.subheader("Diagnostic Plots")
                                st.write("Recursive Residuals")
                                fig, ax = plt.subplots(figsize=(10, 4))
                                recursive_residuals = res.recursive_residuals
                                if len(recursive_residuals) > 0:
                                    ax.plot(df.index[-len(recursive_residuals):], recursive_residuals,
                                           label="Recursive Residuals")
                                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
                                    ax.set_title("Recursive Residuals")
                                    ax.set_xlabel("Time" if not isinstance(df.index, pd.DatetimeIndex) else "Date")
                                    ax.set_ylabel("Residual Value")
                                    ax.legend()
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                else:
                                    st.write("No recursive residuals available.")

                        except Exception as e:
                            st.error(f"Error in recursive least squares analysis: {str(e)}")
                            st.exception(e)

                    # Tab 5: Dominance Analysis
                    with tab5:
                        st.header("Dominance Analysis")
                        st.write("""
                        Dominance analysis helps determine the relative importance of predictor variables by examining 
                        how much each variable contributes to the model's R-squared across all possible variable combinations.
                        """)
                        try:
                            with st.spinner("Running dominance analysis... This may take some time for models with many variables."):
                                dominance_results = dominance_analysis(X, y)
                            st.subheader("Variable Importance (Average Contribution to R²)")
                            st.dataframe(dominance_results.style.format({
                                'Average_Contribution': '{:.6f}'
                            }))
                            fig, ax = plt.subplots(figsize=(10, 6))
                            bars = ax.barh(dominance_results['Variable'], dominance_results['Average_Contribution'],
                                           color='skyblue')
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width + 0.001,
                                        bar.get_y() + bar.get_height() / 2,
                                        f'{width:.4f}',
                                        ha='left',
                                        va='center')
                            ax.set_xlabel('Average Contribution to R²')
                            ax.set_ylabel('Variable')
                            ax.set_title('Variable Importance from Dominance Analysis')
                            plt.tight_layout()
                            st.pyplot(fig)
                            total_importance = dominance_results['Average_Contribution'].sum()
                            dominance_results['Relative_Importance'] = dominance_results['Average_Contribution'] / total_importance * 100
                            st.subheader("Relative Variable Importance (%)")
                            st.dataframe(dominance_results[['Variable', 'Relative_Importance']].style.format({
                                'Relative_Importance': '{:.2f}%'
                            }))
                            fig, ax = plt.subplots(figsize=(10, 10))
                            wedges, texts, autotexts = ax.pie(
                                dominance_results['Relative_Importance'],
                                labels=dominance_results['Variable'],
                                autopct='%1.1f%%',
                                startangle=90,
                                textprops={'fontsize': 9},
                                wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
                            )
                            for autotext in autotexts:
                                autotext.set_fontsize(8)
                                autotext.set_weight('bold')
                            ax.axis('equal')
                            ax.set_title('Relative Variable Importance')
                            plt.tight_layout()
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error in dominance analysis: {str(e)}")
            else:
                st.warning("Please select both dependent and independent variables.")
        else:
            st.error("No numeric columns found in the dataset.")
            st.markdown("<hr><center><small>This app created by Dr Merwan Roudane</small></center>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload an Excel file to begin analysis.")
    st.subheader("Example Data Format")
    st.markdown("""
    Your Excel file should contain:
    - At least one numeric dependent variable
    - One or more numeric independent variables
    - Optionally, a date column for time series analysis

    For best results with time series analyses (rolling regression, recursive least squares), 
    include a date column that can be used as an index.
    """)
