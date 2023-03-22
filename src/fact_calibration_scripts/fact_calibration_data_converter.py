import json
import pandas as pd
import re


# helper to load dataset
def load_full_dataset(
    dataset_filename="calibragpt_full_input_information_3_20_23.csv",
    prefix="../../data/",
):
    return pd.read_csv(prefix + dataset_filename)


# Convert cka fact probing logs to a log of the memit type
# where `log_filename` is the string filename for the particular
# model and runtime log whose outputs we'd like to convert to
# which can be found in (/src/benchmark_scripts/output_logs/)
# which takes the format of:
# [model name]_logged_cka_outputs_[date].json
# the format required by MEMIT (see/notebooks/memit_run_main.ipynb)
def convert_log_to_memit_format(
    log_filename, prefix="../../src/benchmark_scripts/output_logs/", verbose=False
):

    full_df = load_full_dataset()

    try:
        with open(prefix + log_filename, "r") as logdata:
            log_dict = json.load(logdata)

            model_name = log_dict["model_name"][0]

            if model_name is None:
                print(f"Could not extract model name from {log_filename}")
                raise Exception

            if verbose:
                print(f"Converting log for {model_name}")

            score_dict_summary = log_dict["score_dict_summary"][model_name]

            if verbose:
                print(f"score dict summary: {score_dict_summary}")

            n_correct = re.search(r"([0-9]+)\/", score_dict_summary)
            n_total = re.search(r"\/([0-9]+)", score_dict_summary)

            if n_correct is None or n_total is None:
                print(f"could not extract summary performance from {log_filename}")
                raise Exception

            n_correct = n_correct.group(1)
            n_total = n_total.group(1)
            n_wrong = int(n_total) - int(n_correct)
            print(
                f"we should be correcting {n_wrong} incorrect associations via MEMIT."
            )

            score_dict = log_dict["score_dict_full"][model_name]

            if score_dict is None:
                print(f"Encountered issue parsing data for {model_name}")
                raise Exception

            if verbose:
                print(
                    f"Checking {model_name}'s {len(score_dict)} item score_dict for conversions."
                )

            prompts = []
            subjects = []
            targets = []

            facts_to_correct = 0

            for fact_output in score_dict:

                if fact_output["p_true > p_false_average"] == "False":
                    # determine which dataset the entry came from
                    dataset = (
                        "calinet"
                        if fact_output["dataset_original"]
                        == "calinet_input_information"
                        else "rome"
                    )

                    if dataset is None:
                        print(
                            f"Encountered issue resolving original data for {fact_output}"
                        )
                        raise Exception

                    # grab id linking back to original
                    # todo: change this to dataset_id
                    # if we re-run this with new formatted log output, not strictly necessary though
                    id = (
                        fact_output["fact_id"]
                        if dataset == "calinet"
                        else fact_output["case_id"]
                    )

                    if id is None:
                        print(
                            f"Encountered issue resolving original id for {fact_output}"
                        )
                        raise Exception

                    try:
                        # produce the correct prompt and its subject
                        prompt, subject, obj = parse_stem(id, full_df, dataset)

                        # append the prompt
                        prompts.append(prompt)

                        # append the subject
                        subjects.append(subject)

                        # append fact as the target, in ROME format
                        targets.append({"str": obj})

                        # update counter
                        facts_to_correct += 1
                    except Exception:
                        print(f"problem iterating through log file {log_filename}")
                        raise Exception

            if facts_to_correct != n_wrong:
                print("mismatch between model inaccuracies and log-conversion output")
                raise Exception
            else:
                print(
                    f"Conversion complete. Returning {facts_to_correct} associations to correct for {model_name}"
                )
                return pd.DataFrame(
                    {"prompt": prompts, "subject": subjects, "target_new": targets}
                )

    except ValueError:
        print(f"problem decoding log file {log_filename}")
        raise Exception


# helper to split off the entity at the start of the stem
# from the template portion that follows it
def parse_stem(id, df, dataset):

    df_id = "rome_" + str(id) if dataset == "rome" else "calinet_" + str(id)

    # get the row for that id in the df using the dataset info
    row = (
        df[df["dataset_id"] == df_id]
        if dataset == "rome"
        else df[df["dataset_id"] == df_id]
    )

    if row is None:
        print(f"Encountered issue retrieving {df_id} from the full dataset.")
        raise Exception

    # get the subject entity
    subject = list(row["subject"])[0]
    # get the text -- what MEMIT calls the 'prompt'
    prompt = list(row["stem"])[0]
    # get the object -- what is going to get filled in.
    obj = list(row["object"])[0]

    if subject is None or prompt is None or obj is None:
        print(f"Encountered issue gathering data for id {df_id}")
        raise Exception

    # first, check traditional setup
    # where the subject is in the prompt
    if subject in prompt:
        prompt_cleaned = prompt.replace(subject, "{}")

        if prompt == prompt_cleaned or "{}" not in prompt_cleaned:
            print(f"Encountered issue cleaning the prompt {prompt} for {df_id}.")
            raise Exception

        return prompt_cleaned, subject, obj

    else:
        # first: check for case when subject and object should just be swapped
        if obj in prompt:
            prompt_cleaned = prompt.replace(obj, "{}")
            subject_cleaned = obj
            obj_cleaned = subject

            return prompt_cleaned, subject_cleaned, obj_cleaned

        # otherwise, the prompt has the blank at the end
        else:
            prompt_cleaned = prompt + " " + "{}"
            return prompt_cleaned, subject, obj
