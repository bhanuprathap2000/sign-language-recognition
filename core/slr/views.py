from django.shortcuts import render
from .forms import UploadFileForm
from django.views.decorators.csrf import ensure_csrf_cookie
from SLRA.real_time_prediction import get_video_and_predict
@ensure_csrf_cookie
def upload_display_video(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            #print(file.name)
            handle_uploaded_file(file)
            #write the code to predict the sign in video
            prediction_data=get_video_and_predict(f"D:/final year project/SLR/core/{file.name}")

            return render(request, "index.html", {'filename': file.name,"data":prediction_data[0]})
    else:
        form = UploadFileForm()
    return render(request, 'index.html', {'form': form})

def handle_uploaded_file(f):
    with open(f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)