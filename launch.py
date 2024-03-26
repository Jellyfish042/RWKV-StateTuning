from rwkv_model import RWKVInfer
import argparse

INS_TEMPLATE = '''Instruction: <I>\n\nInput: <Q>\n\nResponse: '''
# TEMPLATE = '''Input: <Q>\n\nResponse:'''
TEMPLATE = '''English: <Q>\n\n\Chinese: '''
# TEMPLATE = '''User: <Q>\n\nAssistant: '''


def my_print(t):
    print(t, end='')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--state', type=str)
    parser.add_argument('--instruction', type=str)

    args = parser.parse_args()

    if args.instruction:
        template = INS_TEMPLATE.replace('<I>', args.instruction)
    else:
        template = TEMPLATE

    if args.state:
        model = RWKVInfer(args.model,
                          args.state
                          )
    else:
        model = RWKVInfer(args.model
                          )

    print('Note! the conversation here is single-turn.')
    while True:
        print('\n' + '#' * 100)
        prompt = input('Input:')
        prompt = template.replace('<Q>', prompt)
        print('Output:', end='')
        ids = model.generate(prompt, topp=0.01, stop_token=[0, 11, 261], max_new_tokens=200, recall=my_print)
        # print(ids)
