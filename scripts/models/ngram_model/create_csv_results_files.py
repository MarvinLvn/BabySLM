"""Creating csv files storing the results"""
# python standard libraries imports
from argparse import ArgumentParser
import logging

# installed libraries imports
from tqdm import tqdm
import pandas as pd
from pathlib import Path

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def create_df_by_hours_syntactic(results: Path) -> pd.DataFrame:
    """
    Create a DataFrame storing the results for the syntactic\
    tasks.

    Parameters
    ----------
    - results:
        Path to the the folder storing the results\
        of the syntactic and lexical tasks for all the models
    
    Returns
    -------
    - DataFrame:
        DataFrame of the results.
    """
    all_results = []
    results_types = list(results.rglob("*by_type.csv"))
    for results_type in tqdm(results_types, total=len(results_types)):
        dataset = results_type.parents[0].stem
        model_size = results_type.parents[5].stem
        hour = str(results_type.parents[4])
        hour = hour.split("/")[-1]
        hour = float(hour[:-1])
        training_set = results_type.parents[3].stem
        df = pd.read_csv(results_type)
        df = df.drop(columns=["std"])
        df["Dataset"] = dataset
        df["Model"] = model_size
        df["Hour"] = hour
        df["Training_set"] = training_set
        df["score"] = 100.0 * df.score

        all_results.append(df)

    df = pd.concat(all_results, ignore_index=True)
    grouped = df.groupby(["Dataset", "Model", "Hour", "type"])
    aggregated = grouped.agg({'score': ['mean', 'std']})
    df = aggregated.xs('score', axis=1)
    df = df.rename(columns={"mean":"Score"})
    return df


def create_df_by_hours_lexical(results: Path) -> pd.DataFrame:
    """
    Create a csv file storing the results for the lexical\
    tasks.

    Parameters
    ----------
    - results:
        Path to the the folder storing the results\
        of the syntactic and lexical tasks for all the models
    
    Returns
    -------
    - DataFrame:
        DataFrame of the results.
    """
    all_results = []
    lexical_accuracies = list(results.rglob("*overall_accuracy_lexical*"))
    for results in tqdm(lexical_accuracies, total=len(lexical_accuracies)):
        dataset = results.parents[0].stem
        model_size = results.parents[5].stem
        hour = str(results.parents[4])
        hour = hour.split("/")[-1]
        training_set = results.parents[3].stem
        with open(results, "r") as overall_acc:
            score = next(overall_acc).strip()
        all_results.append({
                            "Dataset" : dataset,
                            "Model": model_size,
                            "Hour": float(hour[:-1]),
                            "score": float(score) * 100.0,
                            "training_set": training_set
                            })
    df = pd.DataFrame(all_results)
    grouped = df.groupby(["Dataset", "Model", "Hour"])
    aggregated = grouped.agg({'score': ['mean', 'std']})
    df = aggregated.xs('score', axis=1)
    df = df.rename(columns={"mean":"Score"})
    return df

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path",
                        help="The folder containing the results of the syntactic and lexical tasks.")
    parser.add_argument("-o", "--output_path",
                        help="The folder where to save the csv files.")
    
    args = parser.parse_args()
    results_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    LOGGER.info("Creating DataFrame for the syntactic tasks")
    syntactic_df = create_df_by_hours_syntactic(results_path)
    LOGGER.info("Creating DataFrame for the lexical tasks")
    lexical_df = create_df_by_hours_lexical(results_path)
    
    syntactic_df.to_csv(output_path / "syntactic_results.csv")
    lexical_df.to_csv(output_path / "lexical_results.csv")


if __name__ == "__main__":
    main()