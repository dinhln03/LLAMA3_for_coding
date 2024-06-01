from recyclus import Client
import time


def load(job):
    print('saving cyclus.sqlite...')
    client.save('cyclus.sqlite', job.jobid)


def wait_for_completion(job):
    while True:
        time.sleep(2)
        resp = job.status()
        if resp['status'] != 'ok':
            print(f'Error:', resp['message'])
            return

        info = resp['info']
        print(f"\tStatus: {info['status']}")

        if info['status'] in ['done', 'error', 'failed', 'unknown job']:
            if info['status'] == 'done':
                load(job)
                # job.delete()
                print('done')
            return


client = Client()
job = client.run(scenario='./scenario.xml', project='demo')

print('job submitted:', job.jobid)
wait_for_completion(job)


print('files:',job.files())

print('list:')
job.list()



