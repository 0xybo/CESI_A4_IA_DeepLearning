from typing import List
import math as math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from lib.display import print_tabs


class Dataset:
    """
    Represente a dataset with its data and target variable. Provides methods for data exploration, 
    cleaning, and visualization.
    """

    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target

    def get_row_count(self) -> int:
        """
        Returns the number of rows in the dataset.

        Returns:
            int: The number of rows in the dataset.
        """
        return len(self.data)

    def drop(self, columns: List[str], inplace: bool = False) -> "Dataset":
        """
        Drops specified columns from the dataset.

        Args:
            columns (List[str]): A list of column names to drop.
            inplace (bool, optional): If True, modifies the dataset in place. If False, returns a 
              new Dataset instance with the specified columns dropped. Defaults to False.

        Returns:
            Dataset: A new Dataset instance with the specified columns dropped if inplace is False, 
            otherwise returns self.
        """
        if inplace:
            self.data.drop(columns=columns, inplace=True)
            return self
        return Dataset(self.data.drop(columns=columns), target=self.target)

    def advanced_describe(self) -> None:
        """
        Provides an advanced description of the dataset, including information, head, description, 
        and correlations.

        This method uses the print_tabs function to display different aspects of the dataset in a 
        tabbed format, allowing for easy exploration of the data. It includes:
        - "Informations": Displays the dataset information using the info() method.
        - "Aperçu": Shows the first few rows of the dataset using the head() method.
        - "Description": Provides a statistical summary of the dataset using the describe() 
          method.
        - "Corrélations": Visualizes the correlations between the features and the target variable 
          using the draw_correlations() method.
        """
        print_tabs(
            {
                "Informations": lambda: self.data.info(),
                "Aperçu": lambda: self.data.head(),
                "Description": lambda: self.data.describe(),
                "Corrélations": lambda: self.draw_correlations(),
            }
        )

    def draw_correlations(self, show: bool = False) -> None:
        """
        Visualizes the correlations between the features and the target variable using a heatmap.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.axes[0]
        correlations = self.data.corr(method="pearson")
        sns.heatmap(correlations, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Corrélations avec la variable cible")
        ax.set_xticks(rotation=45)

        if show:
            plt.show()
        else:
            return fig

    def draw_correlations_with_target(self, show: bool = False) -> None:
        """
        Visualizes the correlations between the features and the target variable using a heatmap.
        This method focuses on the correlations of the features with the target variable, showing 
        only the absolute values of the correlations sorted in descending order. It provides a 
        clear view of which features are most strongly correlated with the target variable, aiding 
        in feature selection and understanding of the dataset.
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.axes[0]
        correlations = (
            self.data.corr(method="pearson")[[self.target]]
            .drop(self.target, axis=0)
            .abs()
            .sort_values(by=self.target, ascending=False)
        )
        sns.heatmap(correlations, annot=True, cmap="coolwarm", ax=ax)
        # sns.barplot(x=correlations.index, y=correlations["target"])
        # plt.xlabel("Variables")
        # plt.ylabel("Corrélation")
        ax.set_title("Corrélations avec la variable cible")
        ax.set_xticks(rotation=45)

        if show:
            plt.show()
        else:
            return fig

    def print_issues(self) -> None:
        """
        Prints the issues in the dataset, including missing values, outliers, categorical 
        variables, and disparities. This method uses the print_tabs function to display
        different types of issues in the dataset in a tabbed format, allowing for easy 
        identification and exploration of potential problems in the data. It includes:
        - "Valeurs manaquantes": Displays the count of missing values in each column using 
          the get_missing_values() method.
        - "Valeurs aberrantes": Shows the count of outliers in each column using the 
          get_outliers() method.
        - "Variables catégorielles": Lists the categorical variables in the dataset using 
          the get_categorical_variables() method.
        - "Disparités": Visualizes the distribution of the target variable to identify any 
          class imbalances using the print_disparity() method.
        """
        print_tabs(
            {
                "Valeurs manaquantes": lambda: self.print_missing_values(),
                "Valeurs aberrantes": lambda: self.print_outliers(),
                "Variables catégorielles": lambda: self.print_categorical_variables(),
                "Disparités": lambda: self.print_disparity(),
            }
        )

    def get_missing_values(self) -> pd.Series:
        """
        Returns a Series containing the count of missing values for each column in the dataset.

        Returns:
            pd.Series: A Series with the count of missing values for each column.
        """
        return self.data.isnull().sum()

    def get_outliers(self, threshold: float = 3.0) -> pd.Series:
        """
        Returns a Series containing the count of outliers for each column in the dataset based 
        on the Z-score method.

        Args:
            threshold (float, optional): The Z-score threshold to identify outliers. 
            Defaults to 3.0.
        Returns:
            pd.Series: A Series with the count of outliers for each column.
        """
        numeric_data = self.data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            return pd.Series(0, index=self.data.columns, dtype="int64")

        std = numeric_data.std().replace(0, np.nan)
        z_scores = (numeric_data - numeric_data.mean()) / std
        outlier_counts_numeric = (np.abs(z_scores) > threshold).sum()

        outlier_counts = pd.Series(0, index=self.data.columns, dtype="int64")
        outlier_counts.loc[outlier_counts_numeric.index] = (
            outlier_counts_numeric.astype("int64")
        )

        return outlier_counts

    def get_categorical_variables(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing the names of categorical variables in the dataset.

        Returns:
            pd.DataFrame: A DataFrame with the names of categorical variables in the dataset.
        """
        return self.data.select_dtypes(include=["object", "category"])

    def draw_disparity(self, show: bool = False) -> None:
        """
        Visualizes the distribution of the target variable to identify any class imbalances.
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.axes[0]
        sns.countplot(x=self.data[self.target], data=self.data, ax=ax)
        ax.set_title("Distribution de la variable cible")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Nombre d'échantillons")

        if show:
            plt.show()
        else:
            return fig

    def get_duplicate_count(self) -> int:
        """
        Returns the number of duplicate rows in the dataset.

        Returns:
            int: The number of duplicate rows in the dataset.
        """
        return self.data.duplicated().sum()

    def drop_duplicates(self) -> None:
        """
        Removes duplicate rows from the dataset.
        """
        self.data = self.data.drop_duplicates()

    def normalize_zscore(self, threshold: int) -> "Dataset":
        """
        Returns the dataset after normalization with the z-score method.

        Args:
            threshold (int): the limit for the z-score, after which the data is dropped.

        Returns:
            Dataset: A Dataset instance containing the remaining data from the Dataset after normalization.
        """
        tmp_zscore = (dataset.data['BMI'] - dataset.data['BMI'].mean()) / dataset.data['BMI'].std()
tmp_zscore_data = dataset.data[tmp_zscore.abs() < 3]
tmp_zscore_dataset = Dataset(tmp_zscore_data, dataset.target)

    def draw_distributions(self, columns: list | None = None, show: bool = False) -> None:
        """
        Visualizes the distribution of all numeric and categorical features in the dataset 
        using histograms for numeric features and count plots for categorical features. 
        All plots are displayed in the same figure.
        """
        if columns is not None:
            data = self.data[columns]
        else:
            data = self.data

        numeric_data = data.select_dtypes(include=[np.number])
        categorical_data = data.select_dtypes(include=["object", "category"])

        num_numeric = len(numeric_data.columns)
        num_categorical = len(categorical_data.columns)
        total_plots = num_numeric + num_categorical
        cols = 3
        rows = math.ceil(total_plots / cols)
        fig = plt.figure(figsize=(cols * 5, rows * 4))
        plot_index = 1
        for column in numeric_data.columns:
            ax = fig.add_subplot(rows, cols, plot_index)
            sns.histplot(numeric_data[column], discrete=True, ax=ax)
            ax.set_title(f"Distribution de {column}")
            plot_index += 1
        for column in categorical_data.columns:
            ax = plt.add_subplot(rows, cols, plot_index)
            sns.countplot(x=categorical_data[column], ax=ax)
            ax.set_title(f"Distribution de {column}")
            ax.set_xticks(rotation=45)
            plot_index += 1
        fig.tight_layout()

        if show:
            plt.show()
        else: 
            return fig

    def draw_distribution(self, column: str, show: bool = False) -> None:
        """
        Visualizes the distribution of a specific feature in the dataset using a histogram 
        for numeric features or a count plot for categorical features.

        Args:
            column (str): The name of the column to visualize.
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found in the dataset.")
            return

        fig = plt.figure(figsize=(8, 6))
        ax = fig.axes[0]
        if pd.api.types.is_numeric_dtype(self.data[column]):
            sns.histplot(self.data[column], discrete=True, ax=ax)
            ax.set_title(f"Distribution de {column}")
        else:
            sns.countplot(x=self.data[column], ax=ax)
            ax.set_title(f"Distribution de {column}")
            ax.set_xticks(rotation=45)
        fig.tight_layout()

        if show:
            plt.show()
        else: 
            return fig

    def draw_boxplots(self, columns: list | None = None, show: bool = False) -> None:
        """
        Visualizes the distribution of numeric features in the dataset using boxplots 
        to identify outliers and understand the spread of the data.
        """
        if columns is not None:
            numeric_data = self.data[columns]
        else:
            numeric_data = self.data

        numeric_data = numeric_data.select_dtypes(include=[np.number])
        num_numeric = len(numeric_data.columns)
        cols = 3
        rows = math.ceil(num_numeric / cols)
        fig = plt.figure(figsize=(cols * 5, rows * 4))
        for index, column in enumerate(numeric_data.columns):
            ax = fig.add_subplot(rows, cols, index + 1)
            sns.boxplot(y=numeric_data[column], ax=ax)
            ax.set_title(f"Boxplot de {column}")
        fig.tight_layout()

        if show:
            plt.show()
        else: 
            return fig

    def draw_boxplot(self, column: str, show: bool = False) -> None:
        """
        Visualizes the distribution of a specific numeric feature in the dataset using 
        a boxplot to identify outliers and understand the spread of the data.

        Args:
            column (str): The name of the numeric column to visualize.
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found in the dataset.")
            return

        if not pd.api.types.is_numeric_dtype(self.data[column]):
            print(
                f"Column '{column}' is not numeric and cannot be visualized with a boxplot."
            )
            return

        fig = plt.figure(figsize=(8, 6))
        ax = fig.axes[0]
        sns.boxplot(y=self.data[column], ax=ax)
        ax.set_title(f"Boxplot de {column}")
        fig.tight_layout()

        if show:
            plt.show()
        else: 
            return fig

    @staticmethod
    def from_csv(src: str, target: str) -> "Dataset":
        """
        Creates a Dataset instance from a CSV file.

        Args:
            src (str): The path to the CSV file.
            target (str): The name of the target variable in the dataset.
        Returns:
            Dataset: A Dataset instance containing the data from the CSV file and the 
            specified target variable.
        """
        data = pd.read_csv(src)
        return Dataset(data, target)
