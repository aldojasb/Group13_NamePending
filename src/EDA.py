"""This script saves EDA charts.
Usage: EDA.py <path_to_x_train> <path_to_y_train> <loc_to_be_saved>

Options:
<path_to_x_train>             path to X_train.csv
<path_to_y_train>             path to y_train.csv
<loc_to_be_saved>             path to where we want to save the images

"""

import altair as alt
from docopt import docopt
import pandas as pd


def main():
    opt = docopt(__doc__)

    x_path = opt['<path_to_x_train>']
    y_path = opt['<path_to_y_train>']
    save_path = opt['<loc_to_be_saved>']
    # print(save_path)

    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path)
    data = pd.concat([X_train, y_train], axis=1)

    alt.renderers.enable('mimetype')

    chart1 = alt.Chart(
        data, title="Distribution of white wine quality"
    ).mark_bar().encode(
        x=alt.X("quality", bin=alt.Bin(maxbins=12), title="Wine quality (/10)"),
        y="count()"
    ).properties(
        width=500,
        height=500
    ).configure_axis(
        labelFontSize=15,
        titleFontSize=20
    ).configure_title(fontSize=20)

    chart1.save(
        f'{save_path}/Distribution_of_white_wine_quality.png',
        scale_factor=2.0
    )

    chart2 = alt.Chart(
        data, title="Relationship between features and target (1/3)"
    ).mark_boxplot(opacity=1, size=10).encode(
        y=alt.Y(alt.repeat(),
                type="quantitative",
                scale=alt.Scale(zero=False)),
        x=alt.X("quality", scale=alt.Scale(zero=False), ),
        color=alt.Color("quality", legend=None)
    ).properties(
        width=300,
        height=600
    ).repeat(
        ["fixed acidity", "volatile acidity", "citric acid", "residual sugar"]
    ).configure_axis(
        labelFontSize=15, titleFontSize=20)

    chart2.save(
        f'{save_path}/relationship_between_individual_features_and_the_quality_1.png',
        scale_factor=2.0)

    chart3 = alt.Chart(
        data, title="Relationship between features and target (2/3)"
    ).mark_boxplot(opacity=1, size=10).encode(
        y=alt.Y(alt.repeat(),
                type="quantitative",
                scale=alt.Scale(zero=False)),
        x=alt.X("quality", scale=alt.Scale(zero=False)),
        color=alt.Color("quality", legend=None)
    ).properties(
        width=300,
        height=600
    ).repeat(
        ["chlorides", "free sulfur dioxide", "total sulfur dioxide", "density"]
    ).configure_axis(
        labelFontSize=15, titleFontSize=20)

    chart3.save(
        f'{save_path}/relationship_between_individual_features_and_the_quality_2.png',
        scale_factor=2.0)

    chart4 = alt.Chart(
        data, title="Relationship between features and target (3/3)"
    ).mark_boxplot(opacity=1, size=10).encode(
        y=alt.Y(alt.repeat(),
                type="quantitative",
                scale=alt.Scale(zero=False)),
        x=alt.X("quality", scale=alt.Scale(zero=False)),
        color=alt.Color("quality", legend=None)
    ).properties(
        width=300,
        height=600
    ).repeat(
        ["pH", "sulphates", "alcohol"]
    ).configure_axis(
        labelFontSize=15, titleFontSize=20)

    chart4.save(
        f'{save_path}/relationship_between_individual_features_and_the_quality_3.png',
        scale_factor=2.0)


if __name__ == "__main__":
    main()
