import os
import threading
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import messagebox, StringVar

import matplotlib.pyplot as plt
from speech import SpeechRecognizer


class MainWindow(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.rec = None
        self.root = root
        self.window_width, self.window_height = 800, 400
        self.root.minsize(800, 400)
        self.root.maxsize(800, 400)
        self.root.title('Распознавание речи')
        root_width, root_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        left = (root_width - self.window_width) // 2
        top = (root_height - self.window_height) // 2
        self.root.geometry(f'{self.window_width}x{self.window_height}+{left}+{top}')

        self.f_contols = tk.Frame(master=self.root, width=self.window_width // 2)
        self.f_indicators = tk.Frame(master=self.root, width=self.window_width // 2)

        self.btn_create = tk.Button(
            self.f_contols,
            text='Выбрать файл\nи запустить распознавание',
            command=self.on_open
        )
        self.n_speakers = StringVar()
        self.n_speakers_input = tk.Spinbox(self.f_contols, from_=0, to=5, increment=1, textvariable=self.n_speakers)
        self.n_speakers_label = tk.Label(self.f_contols, text='Число спикеров (0 - автоподбор):')
        self.btn_export = tk.Button(
            self.f_contols,
            text='Сохранить отчет',
            command=self.on_export
        )
        self.output = tk.Text(self.f_indicators, wrap=tk.WORD)
        self.reset_output()

        self.n_speakers.set('2')

        self.f_contols.pack(side=tk.LEFT, expand=False, fill=tk.BOTH)
        self.f_indicators.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        self.btn_create.pack(side=tk.TOP, padx=10, pady=10)
        self.n_speakers_label.pack(side=tk.TOP, padx=10, pady=10)
        self.n_speakers_input.pack(side=tk.TOP, padx=10, pady=10)
        self.btn_export.pack(side=tk.TOP, padx=10, pady=10)

        self.output.pack(side=tk.TOP, padx=10, pady=10)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def set_output(self, text: str) -> None:
        self.output.config(state=tk.NORMAL)
        self.output.delete('1.0', 'end')
        self.output.insert(tk.END, text)
        self.output.config(state=tk.DISABLED)

    def reset_output(self):
        self.set_output('Здесь будет отображен результат.')

    def on_close(self):
        plt.close()
        self.root.destroy()

    def on_export(self):
        filename = fd.asksaveasfilename(filetypes=[('Text file', '*.txt')])
        if not filename:
            return
        if not filename.endswith('.txt'):
            filename += '.txt'
        with open(filename, 'w') as fout:
            fout.write(self.output.get('1.0', tk.END))

    def on_error(self):
        messagebox.showerror('Что-то пошло не так(((')
        self.reset_output()
        self.on_close()

    def _check_thread(self, thread, message: str, next_action: str, iters: int = 0):
        if thread.is_alive():
            self.set_output(message + '.' * (iters % 4))
            self.after(250, lambda: self._check_thread(thread, message, next_action, iters + 1))
        else:
            if next_action == 's2t':
                self.s2t()
            elif next_action == 'finilize':
                self.finilize()
            else:
                raise ValueError(f'Unknown action "{next_action}"')

    def _unblock_controls(self):
        self.btn_create['state'] = tk.NORMAL
        self.btn_export['state'] = tk.NORMAL
        self.n_speakers_input['state'] = tk.NORMAL

    def _block_controls(self):
        self.btn_create['state'] = tk.DISABLED
        self.btn_export['state'] = tk.DISABLED
        self.n_speakers_input['state'] = tk.DISABLED

    def on_open(self):
        self._block_controls()
        try:
            n_speakers = int(self.n_speakers.get())
            if n_speakers < 0:
                raise ValueError()
        except ValueError:
            messagebox.showerror(
                'Неверный ввод',
                'Число спикеров должно быть целым положительным числом.'
            )
            self._unblock_controls()
            self.reset_output()
            return
        ffmpeg_path = fd.askopenfilename(title='Укажите путь к ffmpeg', filetypes=[('Executable',  '*.exe')])
        if not ffmpeg_path:
            self._unblock_controls()
            return
        filename = fd.askopenfilename(filetypes=[('mp3 audio',  '*.mp3'), ('WAV', '*.wav')])
        if not filename:
            self._unblock_controls()
            return
        self.rec = SpeechRecognizer(filename, n_speakers, ffmpeg_path)
        try:
            if not os.path.exists(self.rec.wavfile):
                self.set_output('Конвертируем файл в WAV...')
                self.rec.convert_to_wav()
        except Exception:  # noqa
            self.on_error()
        try:
            diar_thread = threading.Thread(target=self.rec.diarize, name="Diarization")
            diar_thread.start()
            self._check_thread(diar_thread, 'Распознаем спикеров', 's2t')
        except Exception:  # noqa
            self.on_error()

    def s2t(self):
        try:
            s2t_thread_thread = threading.Thread(target=self.rec.speech_to_text, name="Speech2Text")
            s2t_thread_thread.start()
            self._check_thread(s2t_thread_thread, 'Преобразуем в текст', 'finilize')
        except Exception:  # noqa
            self.on_error()

    def finilize(self):
        self.set_output('Собираем ответ...')
        self.root.update()
        res = 'Что-то пошло не так((('
        try:
            res = self.rec.finilize()
        except Exception:  # noqa
            self.on_error()
        print(res)
        self.set_output(res)
        self._unblock_controls()


if __name__ == '__main__':
    rootObj = tk.Tk()
    app = MainWindow(rootObj)
    app.pack()
    rootObj.mainloop()
