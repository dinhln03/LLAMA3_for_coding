import PySimpleGUI as sg

layout = [
  [sg.Text('text')],
  [sg.Input('input', key= 'input1')],
  [sg.Input('input', key='input2')],
  [sg.Button('button', key='button1')]
]

window = sg.Window('list values - list or dict', layout)

while True:
  event, values = window.Read()

  if event == 'button1':
    print(values['input1'])
    print(values['input2'])

    # prints button key because that's current events' key
    print(event)

  elif event is None:
    break


window.Close()
