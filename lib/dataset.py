from typing import List, Literal
import math as math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
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

    def draw_correlations(
        self,
        ax: Axes | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        show: bool = True,
    ) -> None:
        """
        Visualizes the correlations between the features and the target variable using a heatmap.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot()
        else:
            fig = ax.figure
        correlations = self.data.corr(method=method)
        sns.heatmap(correlations, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Corrélations avec la variable cible")
        ax.tick_params("x", rotation=45)

        if show:
            fig.tight_layout()
            plt.show()

    def draw_correlations_with_target(
        self,
        ax: Axes | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        show: bool = True,
    ) -> None:
        """
        Visualizes the correlations between the features and the target variable using a heatmap.
        This method focuses on the correlations of the features with the target variable, showing
        only the absolute values of the correlations sorted in descending order. It provides a
        clear view of which features are most strongly correlated with the target variable, aiding
        in feature selection and understanding of the dataset.
        """
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot()
        else:
            fig = ax.figure
        correlations = (
            self.data.corr(method=method)[[self.target]]
            .drop(self.target, axis=0)
            .abs()
            .sort_values(by=self.target, ascending=False)
        )
        sns.heatmap(correlations, annot=True, cmap="coolwarm", ax=ax)
        # sns.barplot(x=correlations.index, y=correlations["target"])
        # plt.xlabel("Variables")
        # plt.ylabel("Corrélation")
        ax.set_title("Corrélations avec la variable cible")
        ax.tick_params("x", rotation=45)

        if show:
            fig.tight_layout()
            plt.show()

    def normalize_minmax(self, columns: list[str]) -> "Dataset":
        """
        Returns the dataset after normalization with the min-max method.

        Args:
            column (str): the column to normalize.

        Returns:
            Dataset: A Dataset instance containing the remaining data from the Dataset after normalization.
        """
        for column in columns:
            min_val = self.data[column].min()
            max_val = self.data[column].max()
            normalized_data = self.data.copy(deep=True)
            normalized_data[column] = (normalized_data[column] - min_val) / (
                max_val - min_val
            )
        return Dataset(normalized_data, self.target)

    def normalize_zscore(self, columns: list[str]) -> "Dataset":
        """
        Returns the dataset after normalization with the z-score method.

        Returns:
            Dataset: A Dataset instance containing the remaining data from the Dataset after normalization.
        """
        normalized_data = self.data.copy(deep=True)
        for column in columns:
            zscore = (
                normalized_data[column] - normalized_data[column].mean()
            ) / normalized_data[column].std()
            normalized_data[column] = zscore
        return Dataset(normalized_data, self.target)

    def normalize_rubust_scaling(self, columns: list[str]) -> "Dataset":
        """
        Returns the dataset after normalization with the robust scaling method.

        Returns:
            Dataset: A Dataset instance containing the remaining data from the Dataset after normalization.
        """
        normalized_data = self.data.copy(deep=True)
        for column in columns:
            median = normalized_data[column].median()
            mad = (normalized_data[column] - median).abs().median()
            normalized_data[column] = (
                (normalized_data[column] - median) / mad if mad != 0 else 0
            )
        return Dataset(normalized_data, self.target)

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Returns the first n rows of the dataset.

        Args:
            n (int, optional): The number of rows to return. Defaults to 5.

        Returns:
            pd.DataFrame: A DataFrame containing the first n rows of the dataset.
        """
        return self.data.head(n)

    def info(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset, including the number of non-null values and data types for each column.
        """
        columns = ["Column", "Non-Null Count", "Data Type"]
        rows = [
            [col, self.data[col].notnull().sum(), self.data[col].dtype]
            for col in self.data.columns
        ]
        return pd.DataFrame(rows, columns=columns)

    def describe(self) -> pd.DataFrame:
        """
        Returns a DataFrame containing summary statistics for the numeric columns in the dataset.

        Returns:
            pd.DataFrame: A DataFrame with summary statistics for the numeric columns in the dataset.
        """
        return self.data.describe()

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

    def draw_disparity(self, ax: Axes | None = None) -> None:
        """
        Visualizes the distribution of the target variable to identify any class imbalances.
        """
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        else:
            fig = ax.figure
        sns.countplot(x=self.data[self.target], data=self.data, ax=ax)
        ax.set_title("Distribution de la variable cible")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Nombre d'échantillons")

        plt.show()

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

    def filter_outliers_zscore(self, column: str, threshold: int = 3) -> "Dataset":
        """
        Returns the dataset after filtering with the z-score method.

        Args:
            threshold (int): the limit for the z-score, after which the data is dropped.
            column (str): the column to filter.

        Returns:
            Dataset: A Dataset instance containing the remaining data from the Dataset after filtering.
        """
        zscore = (self.data[column] - self.data[column].mean()) / self.data[
            column
        ].std()
        zscore_data = self.data[zscore.abs() < threshold]
        return Dataset(zscore_data, self.target)

    def filter_outliers_iqr(self, column: str) -> "Dataset":
        """
        Returns the dataset after filtering with the IQR method.

        Args:
            column (str): the column to filter.

        Returns:
            Dataset: A Dataset instance containing the remaining data from the Dataset after filtering.
        """
        q1 = self.data[column].quantile(0.25)
        q3 = self.data[column].quantile(0.75)
        iqr = q3 - q1
        iqr_data = self.data[
            (self.data[column] >= q1 - 1.5 * iqr)
            & (self.data[column] <= q3 + 1.5 * iqr)
        ]
        return Dataset(iqr_data, self.target)

    def draw_distributions(
        self, columns: list | None = None, ax: list[Axes] | np.ndarray | None = None
    ) -> None:
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
        if ax is None:
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes_flat = np.atleast_1d(axes).flatten()
        else:
            axes_flat = np.atleast_1d(ax).flatten()
            if len(axes_flat) < total_plots:
                raise ValueError(
                    f"Expected at least {total_plots} axes, got {len(axes_flat)}."
                )
            fig = axes_flat[0].figure

        plot_index = 1
        for column in numeric_data.columns:
            current_ax = axes_flat[plot_index - 1]
            sns.histplot(numeric_data[column], discrete=True, ax=current_ax)
            current_ax.set_title(f"Distribution de {column}")
            plot_index += 1
        for column in categorical_data.columns:
            current_ax = axes_flat[plot_index - 1]
            sns.countplot(x=categorical_data[column], ax=current_ax)
            current_ax.set_title(f"Distribution de {column}")
            current_ax.tick_params("x", rotation=45)
            plot_index += 1

        for index in range(total_plots, len(axes_flat)):
            axes_flat[index].set_visible(False)

        fig.tight_layout()

        plt.show()

    def draw_distribution(
        self,
        column: str,
        ax: Axes | None = None,
        show: bool = True,
        discrete: bool = True,
        bins: int = 30
    ) -> None:
        """
        Visualizes the distribution of a specific feature in the dataset using a histogram
        for numeric features or a count plot for categorical features.

        Args:
            column (str): The name of the column to visualize.
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found in the dataset.")
            return

        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        else:
            fig = ax.figure

        if pd.api.types.is_numeric_dtype(self.data[column]):
            if discrete:
                sns.histplot(self.data[column], discrete=True, ax=ax)
            else:
                sns.histplot(self.data[column], discrete=False, ax=ax, bins=bins)
            ax.set_title(f"Distribution de {column}")
        else:
            sns.countplot(x=self.data[column], ax=ax)
            ax.set_title(f"Distribution de {column}")
            ax.tick_params("x", rotation=45)

        if show:
            fig.tight_layout()
            plt.show()
        else:
            return ax

    def draw_boxplots(
        self, columns: list | None = None, ax: list[Axes] | np.ndarray | None = None
    ) -> None:
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
        if ax is None:
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
            axes_flat = np.atleast_1d(axes).flatten()
        else:
            axes_flat = np.atleast_1d(ax).flatten()
            if len(axes_flat) < num_numeric:
                raise ValueError(
                    f"Expected at least {num_numeric} axes, got {len(axes_flat)}."
                )
            fig = axes_flat[0].figure

        for index, column in enumerate(numeric_data.columns):
            current_ax = axes_flat[index]
            sns.boxplot(y=numeric_data[column], ax=current_ax)
            current_ax.set_title(f"Boxplot de {column}")

        for index in range(num_numeric, len(axes_flat)):
            axes_flat[index].set_visible(False)

        fig.tight_layout()

        plt.show()

    def draw_boxplot(self, column: str, ax: Axes | None = None) -> None:
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

        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot()
        else:
            fig = ax.figure

        sns.boxplot(y=self.data[column], ax=ax)
        ax.set_title(f"Boxplot de {column}")
        fig.tight_layout()

        plt.show()

    def clone(self) -> "Dataset":
        """
        Creates a deep copy of the Dataset instance.

        Returns:
            Dataset: A new Dataset instance that is a deep copy of the original.
        """
        return Dataset(self.data.copy(deep=True), self.target)

    def one_hot_encode(self, columns: list, inplace: bool = False) -> "Dataset":
        """
        Returns a new Dataset instance with the specified categorical columns one-hot encoded.

        Args:
            columns (list): A list of column names to one-hot encode.
            inplace (bool): Whether to modify the dataset in place.

        Returns:
            Dataset: A new Dataset instance with the specified columns one-hot encoded.
        """
        data_encoded = self.data.copy(deep=True)
        for column in columns:
            categories = data_encoded[column].unique()
            for category in categories:
                new_column_name = f"{column}_{category}"
                data_encoded[new_column_name] = (
                    data_encoded[column] == category
                ).astype(int)
            data_encoded.drop(columns=[column], inplace=True)

        if inplace:
            self.data = data_encoded
            return self
        else:
            return Dataset(data_encoded, target=self.target)

    def export_to_csv(self, dest: str = "dataset/output", train: int = 70, test: int = 15) -> None:
        """
        Exports the dataset to CSV files using an optional stratified split on the target.

        Args:
            dest (str, optional): Base path for output CSV files. Defaults to "dataset/output".
            train (int, optional): Percentage of rows to include in the training set. Defaults to 70.
            test (int, optional): Percentage of rows to include in the testing set. Defaults to 15.

        The remaining part becomes validation set.
        Le jeu de train est stratifié sur la colonne cible et rééquilibré en 50/50 si la colonne se nomme Diabetes_binary.
        """

        if self.target not in self.data.columns:
            raise ValueError(f"Target column '{self.target}' introuvable dans le DataFrame.")

        if not (0 <= train <= 100 and 0 <= test <= 100):
            raise ValueError("train et test doivent être des pourcentages entre 0 et 100")

        validation = 100 - (train + test)
        if validation < 0:
            raise ValueError("La somme de train et test ne peut pas dépasser 100.")

        data = self.data.copy(deep=True)

        if self.target == "Diabetes_binary":
            positives = data[data[self.target] == 1]
            negatives = data[data[self.target] == 0]
            if positives.empty or negatives.empty:
                raise ValueError("Impossible de stratifier : pas de classes 0 ou 1 pour Diabetes_binary")

            n_train_per_class = min(
                int(len(positives) * train / 100),
                int(len(negatives) * train / 100),
            )

            train_pos = positives.sample(n=n_train_per_class, random_state=42)
            train_neg = negatives.sample(n=n_train_per_class, random_state=42)
            train_data = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42)

            remaining = data.drop(train_data.index)
        else:
            train_data = data.sample(frac=train / 100, random_state=42)
            remaining = data.drop(train_data.index)

        test_data = remaining.sample(frac=(test / (test + validation)) if (test + validation) > 0 else 0, random_state=42) if test > 0 else pd.DataFrame(columns=data.columns)
        validation_data = remaining.drop(test_data.index) if validation > 0 else pd.DataFrame(columns=data.columns)

        if not train_data.empty:
            train_data.to_csv(dest + "_train.csv", index=False)
        if not test_data.empty:
            test_data.to_csv(dest + "_test.csv", index=False)
        if not validation_data.empty:
            validation_data.to_csv(dest + "_validation.csv", index=False)

        # Affiche la répartition des ensembles via Matplotlib
        repartition = {
            "train": len(train_data),
            "test": len(test_data),
            "validation": len(validation_data),
        }

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(repartition.keys(), repartition.values(), color=["tab:blue", "tab:orange", "tab:green"])
        ax.set_title("Répartition des données après split")
        ax.set_ylabel("Nombre de lignes")
        ax.bar_label(bars)
        ax.set_ylim(0, max(repartition.values()) * 1.1 if repartition.values() else 1)
        plt.tight_layout()
        plt.show()

        # Affiche la distribution des classes dans le train si Diabetes_binary
        if self.target == "Diabetes_binary" and not train_data.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.countplot(x=train_data[self.target], ax=ax2, palette=["tab:red", "tab:blue"])
            ax2.set_title("Distribution des classes dans le train (0: Non-diabétique, 1: Diabétique)")
            ax2.set_xlabel("Classe")
            ax2.set_ylabel("Nombre d'échantillons")
            ax2.bar_label(ax2.containers[0])
            plt.tight_layout()
            plt.show()

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(x=test_data[self.target], ax=ax2, palette=["tab:red", "tab:blue"])
        ax2.set_title("Distribution des classes dans le test (0: Non-diabétique, 1: Diabétique)")
        ax2.set_xlabel("Classe")
        ax2.set_ylabel("Nombre d'échantillons")
        ax2.bar_label(ax2.containers[0])
        plt.tight_layout()
        plt.show()
        
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(x=validation_data[self.target], ax=ax2, palette=["tab:red", "tab:blue"])
        ax2.set_title("Distribution des classes dans le validation (0: Non-diabétique, 1: Diabétique)")
        ax2.set_xlabel("Classe")
        ax2.set_ylabel("Nombre d'échantillons")
        ax2.bar_label(ax2.containers[0])
        plt.tight_layout()
        plt.show()
        
    def to_csv(self, dest: str = "dataset/output", ) -> None:
        """
        Saves the dataset to a CSV file.

        Args:
            dest (str, optional): The path to the output CSV file. Defaults to "output.csv".
        """
        self.data.to_csv(dest + ".csv", index=False)
    
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
