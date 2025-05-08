from datetime import datetime, timedelta

import numpy as _np  # Explicitly use _np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Configuration for Example Data ---
EXAMPLE_TICKER_SYMBOL = "EXAMPLE_AAPL"
EXAMPLE_CURRENT_PRICE = 170.0
EXAMPLE_BASE_IV = 0.20  # 20%
EXAMPLE_SMILE_CURVATURE = 0.5  # Higher values = more smile
EXAMPLE_TERM_SLOPE = 0.05  # How IV changes with TTE (e.g., 0.05 means IV increases by 5% per year of TTE)
EXAMPLE_SKEW_FACTOR = (
    -0.1
)  # Negative for typical equity skew (puts more expensive than equidistant calls)

# --- Helper Functions ---


# NEW: Function to generate example options data
def generate_example_options_data(current_price):
    """
    Generates a synthetic DataFrame of options data for a volatility surface.
    """
    all_options_data = []
    today = datetime.now().date()

    # Define TTEs (in years)
    ttes_years = _np.array(
        [
            30 / 365.25,
            60 / 365.25,
            90 / 365.25,
            180 / 365.25,
            270 / 365.25,
            365 / 365.25,
            540 / 365.25,
        ]
    )

    # Define strikes as a range around the current price
    strike_multipliers = _np.array(
        [0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20]
    )
    strikes = current_price * strike_multipliers

    for tte in ttes_years:
        expiry_date = today + timedelta(days=int(tte * 365.25))
        expiry_str = expiry_date.strftime("%Y-%m-%d")

        for strike_val in strikes:
            # Synthetic IV calculation
            # Basic ATM IV with term structure
            atm_iv_for_tte = EXAMPLE_BASE_IV + EXAMPLE_TERM_SLOPE * tte

            # Smile component (quadratic)
            moneyness = strike_val / current_price
            smile_effect = EXAMPLE_SMILE_CURVATURE * (moneyness - 1) ** 2

            # Skew component (linear to moneyness deviation from ATM)
            skew_effect = EXAMPLE_SKEW_FACTOR * (moneyness - 1)

            iv = atm_iv_for_tte + smile_effect + skew_effect
            iv = max(0.05, iv)  # Ensure IV is not too low or negative

            # Create data for both Calls and Puts (often similar IV surface, but can differ)
            for opt_type in ["Call", "Put"]:
                all_options_data.append(
                    {
                        "ExpirationStr": expiry_str,
                        "ExpirationDate": expiry_date,
                        "TTE_Years": tte,
                        "Strike": strike_val,
                        "ImpliedVolatility": iv,
                        "Type": opt_type,
                        "Volume": 100,  # Dummy data
                        "OpenInterest": 500,  # Dummy data
                        "LastPrice": iv * 0.1 * strike_val,  # Very rough dummy price
                    }
                )

    df = pd.DataFrame(all_options_data)
    return df, current_price


@st.cache_data(ttl=3600 * 24)  # Cache for a day, as it's static example
def load_static_options_data():
    """
    Wrapper function to call the example data generator.
    This keeps the @st.cache_data decorator pattern.
    """
    st.info("Loading example volatility surface data (synthetic).")
    return generate_example_options_data(EXAMPLE_CURRENT_PRICE)


def create_vol_surface_plot(
    df, option_type="Calls", y_axis_type="Strike", current_price=None
):
    """
    Creates a 3D volatility surface plot. (No changes needed here from previous version)
    """
    df_type = df[df["Type"] == option_type.rstrip("s")]

    if df_type.empty:
        st.warning(
            f"No {option_type} data available after filtering to create the plot."
        )
        return go.Figure()

    # Make a copy to avoid SettingWithCopyWarning when adding 'Moneyness'
    df_type_plot = df_type.copy()

    if y_axis_type == "Moneyness" and current_price:
        df_type_plot.loc[:, "Moneyness"] = df_type_plot["Strike"] / current_price
        y_col = "Moneyness"
        y_axis_title = "Moneyness (Strike / Current Price)"
    else:
        y_col = "Strike"
        y_axis_title = "Strike Price"

    try:
        pivot_df = pd.pivot_table(
            df_type_plot,
            values="ImpliedVolatility",
            index=y_col,
            columns="TTE_Years",
            aggfunc="mean",  # Should not be strictly necessary with synthetic data if unique
        )
    except Exception as e:
        st.error(f"Error creating pivot table: {e}")
        st.dataframe(df_type_plot.head())
        return go.Figure()

    if pivot_df.empty:
        st.warning(f"Pivot table for {option_type} is empty. Cannot generate surface.")
        return go.Figure()

    fig = go.Figure(
        data=[
            go.Surface(
                z=pivot_df.values,
                x=pivot_df.columns,  # TTE_Years
                y=pivot_df.index,  # Strike or Moneyness
                colorscale="Viridis",
                colorbar=dict(title="Implied Volatility"),
                contours_z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="limegreen",
                    project_z=True,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"{EXAMPLE_TICKER_SYMBOL} {option_type} Implied Volatility Surface (Example Data)",
        scene=dict(
            xaxis_title="Time to Expiration (Years)",
            yaxis_title=y_axis_title,
            zaxis_title="Implied Volatility (%)",
            xaxis=dict(autorange="reversed"),
            yaxis=dict(
                autorange="reversed"
                if y_axis_type == "Moneyness" and option_type == "Calls"
                else True
            ),
        ),
        margin=dict(l=40, r=40, b=40, t=80),
        height=700,
    )
    fig.update_traces(zhoverformat=".2%")

    if y_axis_type == "Strike" and current_price is not None and not pivot_df.empty:
        x_line = pivot_df.columns.values
        y_line_val = current_price

        z_line_interp = []
        for tte_val in x_line:
            col_data = pivot_df[tte_val].dropna()
            if col_data.empty:
                z_line_interp.append(_np.nan)
                continue

            strikes_for_tte = col_data.index.values
            ivs_for_tte = col_data.values

            sort_indices = _np.argsort(strikes_for_tte)
            sorted_strikes = strikes_for_tte[sort_indices]
            sorted_ivs = ivs_for_tte[sort_indices]

            try:
                interp_iv = _np.interp(
                    current_price,
                    sorted_strikes,
                    sorted_ivs,
                    left=_np.nan,
                    right=_np.nan,
                )
                z_line_interp.append(interp_iv)
            except Exception:
                z_line_interp.append(_np.nan)

        z_line_data = _np.array(z_line_interp)

        valid_mask = ~_np.isnan(z_line_data)
        if _np.any(valid_mask):
            # y_line needs to be an array of the same shape as x_line[valid_mask]
            y_line_plot = _np.full_like(x_line[valid_mask], y_line_val, dtype=float)
            fig.add_trace(
                go.Scatter3d(
                    x=x_line[valid_mask],
                    y=y_line_plot,
                    z=z_line_data[valid_mask],
                    mode="lines",
                    line=dict(color="red", width=5),
                    name=f"Current Price ({current_price:.2f})",
                )
            )
        else:
            st.info(
                "Could not draw current price line due to insufficient data for interpolation at current price level."
            )
    return fig


# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title(f"{EXAMPLE_TICKER_SYMBOL} Volatility Surface Visualizer (Example Data)")
st.markdown(
    "This app visualizes a *synthetically generated* volatility surface, not live market data."
)


st.sidebar.header("Plot Controls")
selected_option_type = st.sidebar.selectbox("Option Type", ["Calls", "Puts"], index=0)
selected_y_axis = st.sidebar.selectbox("Y-Axis", ["Strike", "Moneyness"], index=0)

# Load static example data
options_df, current_stock_price = load_static_options_data()  # Changed this line

if options_df.empty:
    st.error("Example options data is empty. This should not happen.")
else:
    st.success(f"Successfully loaded {len(options_df)} example option contracts.")
    if current_stock_price is not None:
        st.info(f"Example {EXAMPLE_TICKER_SYMBOL} Price: ${current_stock_price:.2f}")

    fig = create_vol_surface_plot(
        options_df, selected_option_type, selected_y_axis, current_stock_price
    )
    st.plotly_chart(fig, use_container_width=True)

    st.sidebar.subheader("Data Preview")
    if st.sidebar.checkbox("Show Example Data Sample for Plot"):
        st.subheader(f"Sample of Example {selected_option_type} Data Used for Plot")

        df_to_show = options_df[options_df["Type"] == selected_option_type.rstrip("s")]
        if selected_y_axis == "Moneyness" and current_stock_price is not None:
            df_to_show_copy = df_to_show.copy()  # Avoid SettingWithCopyWarning
            df_to_show_copy.loc[:, "Moneyness"] = (
                df_to_show_copy["Strike"] / current_stock_price
            )
            st.dataframe(df_to_show_copy.head(10))
        else:
            st.dataframe(df_to_show.head(10))

    if st.sidebar.checkbox("Show All Example Options Data"):
        st.subheader("All Generated Example Options Data")
        st.dataframe(options_df)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"""
    **Example Data Parameters:**
    - Base IV: {EXAMPLE_BASE_IV * 100:.1f}%
    - Smile Curvature: {EXAMPLE_SMILE_CURVATURE}
    - Term Slope: {EXAMPLE_TERM_SLOPE * 100:.1f}% per year
    - Skew Factor: {EXAMPLE_SKEW_FACTOR}
    """)
    st.sidebar.markdown(
        f"Example data generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
