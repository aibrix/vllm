"""
Benchmark the efficiency of prefix caching.

This script allows you to benchmark the performance of
a model with and without prefix caching using either fixed prompts
or prompts sampled from the ShareGPT dataset.

Fixed example usage:
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --num-prompts 1 \
        --repeat-count 100

ShareGPT example usage:
    # This command samples 20 prompts with input lengths
    # between 128 and 256 tokens from the ShareGPT dataset,
    # then replicates each prompt 5 times.
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --enable-prefix-caching \
        --num-prompts 20 \
        --repeat-count 5 \
        --input-length-range 128:256
"""

import json
import random
import time
import os
from typing import List, Optional, Tuple

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.utils import FlexibleArgumentParser

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

# PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"  # noqa: E501
PROMPT = "You are an AI programming assistant and your task is to generate a SQL query based on the input database schema and user questions.\n### Task Description:\nGiven the following database schema, please write a SQL query to answer the given question.\n\n### Schema:\nThe database contains 3 tables: ['customer', 'branch', 'customer_order'].\n\n- **Table**: customer\n\t- *Description*: The table customer has 5 columns: ['Customer_ID', 'Name', 'Nationality', 'Card_Credit', 'Level_of_Membership'].\n\t- *Primary Key*: Customer_ID\n\t- *Foreign Keys*: \n\t- *Column*: Customer_ID\n\t\t- Type: INT\n\t\t- Sampled Values: 1, 2, 3, 4, 5\n\t- *Column*: Name\n\t\t- Type: TEXT\n\t\t- Sampled Values: Arthur Morris, Bill Edrich, Cyril Washbrook, Denis Compton, Donald Bradman\n\t- *Column*: Nationality\n\t\t- Type: TEXT\n\t\t- Sampled Values: Australia, England\n\t- *Column*: Card_Credit\n\t\t- Type: REAL\n\t\t- Sampled Values: 31.9, 42.75, 44.28, 50.85, 62.44\n\t- *Column*: Level_of_Membership\n\t\t- Type: INT\n\t\t- Sampled Values: 0, 1, 2, 3\n\n- **Table**: branch\n\t- *Description*: The table branch has 4 columns: ['Branch_ID', 'Manager', 'Years_opened', 'Location_of_office'].\n\t- *Primary Key*: Branch_ID\n\t- *Foreign Keys*: \n\t- *Column*: Branch_ID\n\t\t- Type: INT\n\t\t- Sampled Values: 1, 2, 3, 4, 5\n\t- *Column*: Manager\n\t\t- Type: TEXT\n\t\t- Sampled Values: Ashby Lazale, Breton Robert, Campbell Jessie, Cobb Sedrick, Hayes Steven\n\t- *Column*: Years_opened\n\t\t- Type: INT\n\t\t- Sampled Values: 2, 3, 4, 5, 6\n\t- *Column*: Location_of_office\n\t\t- Type: TEXT\n\t\t- Sampled Values: Bridgeport, Cheshire, Hartford, Waterbury\n\n- **Table**: customer_order\n\t- *Description*: The table customer_order has 4 columns: ['Customer_ID', 'Branch_ID', 'Dish_Name', 'Quantity'].\n\t- *Primary Key*: ['Customer_ID', 'Branch_ID', 'Dish_Name']\n\t- *Foreign Keys*: customer_order.Customer_ID = customer.Customer_ID, customer_order.Branch_ID = branch.Branch_ID\n\t- *Column*: Customer_ID\n\t\t- Type: INT\n\t\t- Sampled Values: 1, 2, 3, 4, 5\n\t- *Column*: Branch_ID\n\t\t- Type: INT\n\t\t- Sampled Values: 6, 9, 10\n\t- *Column*: Dish_Name\n\t\t- Type: TEXT\n\t\t- Sampled Values: Chow Mein, Kung Pao Chicken, Ma Po Tofu, Peking Roasted Duck, Spring Rolls\n\t- *Column*: Quantity\n\t\t- Type: INT\n\t\t- Sampled Values: 1, 2, 4\n\n### Requirements:\n* Please first return the SQL query to answer the question and then explain your SQL query step by step.\n* Please generate your response using the following format:\n```sql\n<YOUR SQL QUERY>\n```\n<YOUR EXPLANATION>, where the SQL query is in a Markdown code block.\n* Provide a detailed explanation that reflects your reasoning process step by step. Specifically, explain each part of the SQL query (each clause, operator, etc.) and how they work together to answer the question step by step.\n* Please organize the explanation of each SQL step using a Markdown list, following this format:\n- <what is done in the 1st step>\n`<SQL clause used in this step>`\n- <what is done in the 2nd step>\n`<SQL clause used in this step>`\n- <what is done in the 3rd step>\n`<SQL clause used in this step>`...\n* If a certain step involves a nested subquery, provide a detailed explanation for each part of the subquery. You can explain the subquery using the following format:\n- <what is done in the sub-SQL>\n`<sub-SQL>`\n\t* <what is done in the 1st step in sub-SQL>\n\t`<SQL clause used in this step>`\n\t* <what is done in the 2nd step in sub-SQL>\n\t`<SQL clause used in this step>`...\n* When quoting parts of the SQL query in your explanation, please enclose the statement in single backticks, like this: `<part of SQL>`.\n* Please keep your explanation concise and clear within 100 words.\n* Please do NOT select extra columns that are not explicitly requested in the query.\n* Ensure that the table and column names in the generated query exactly match those in the schema. Do NOT include any columns or tables that are not present in the provided schema.\n* Please ensure that the SQL query remains concise and avoids unnecessary joins with unrelated tables.\n\n### Question:\nShow the most common nationality of customers.\n\n### Output:\n"


def test_prefix(llm=None, sampling_params=None, prompts=None):
    start_time = time.time()

    llm.generate(prompts, sampling_params=sampling_params)

    end_time = time.time()
    print(f"cost time {end_time - start_time}")


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: Tuple[int, int],
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    min_len, max_len = input_length_range

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if min_len <= prompt_len <= max_len:
            filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset


def repeat_and_sort_requests(requests: List[Tuple[str, int, int]],
                             repeat_count: int,
                             sort: bool = False) -> List[str]:
    repeated_requests = requests * repeat_count
    if sort:
        repeated_requests.sort(key=lambda x: x[1])
    else:
        random.shuffle(repeated_requests)
    return [req[0] for req in repeated_requests]


def main(args):
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    input_length_range = tuple(map(int, args.input_length_range.split(':')))

    if args.dataset_path is not None:
        print(f"Start to sample {args.num_prompts} prompts"
              "from {args.dataset_path}")
        filtered_datasets = sample_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
        )
    else:
        prompt_len = len(tokenizer(PROMPT).input_ids)
        filtered_datasets = [(PROMPT, prompt_len, args.output_len)
                             ] * args.num_prompts

    llm = LLM(model=args.model,
              tokenizer_mode='auto',
              trust_remote_code=True,
              enforce_eager=True,
              use_v2_block_manager=args.use_v2_block_manager,
              tensor_parallel_size=args.tensor_parallel_size,
              enable_prefix_caching=args.enable_prefix_caching,
              enable_chunked_prefill=args.enable_chunked_prefill,
              max_num_seqs = args.max_num_seqs,
              otlp_traces_endpoint=os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"],
              collect_detailed_traces="all",
            )

    sampling_params = SamplingParams(temperature=0, max_tokens=args.output_len)

    print("Testing filtered datasets")
    prompts = repeat_and_sort_requests(filtered_datasets,
                                       repeat_count=args.repeat_count,
                                       sort=args.sort)

    print("------warm up------")
    test_prefix(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )

    print("------start generating------")
    test_prefix(
        llm=llm,
        prompts=prompts,
        sampling_params=sampling_params,
    )


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description=
        'Benchmark the performance with or without automatic prefix caching.')
    parser.add_argument('--model',
                        type=str,
                        default='baichuan-inc/Baichuan2-13B-Chat')
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--output-len', type=int, default=10)
    parser.add_argument('--enable-prefix-caching',
                        action='store_true',
                        help='enable prefix caching')
    parser.add_argument('--enable-chunked-prefill',
                        action='store_true',
                        help='enable chunked prefill')
    parser.add_argument('--use-v2-block-manager',
                        action='store_true',
                        help='Use BlockSpaceMangerV2')
    parser.add_argument('--num-prompts',
                        type=int,
                        default=1,
                        help="Number of the prompts sampled from dataset")
    parser.add_argument('--repeat-count',
                        type=int,
                        default=100,
                        help='Number of times to repeat each prompt')
    parser.add_argument('--sort',
                        action='store_true',
                        help='Sort prompts by input length')
    parser.add_argument('--input-length-range',
                        type=str,
                        default='128:256',
                        help='Range of input lengths for sampling prompts,'
                        'specified as "min:max" (e.g., "128:256").')
    parser.add_argument('--max-num-seqs', 
                        type=int, 
                        default=256,
                        help='Maximum number of sequences per iteration.')
    args = parser.parse_args()
    main(args)
