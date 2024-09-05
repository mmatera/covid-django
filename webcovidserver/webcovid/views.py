# Create your views here.
# views.py
from django.shortcuts import render
from django.http import JsonResponse
from django.shortcuts import render
from . import models


def index(request):
    # Initial data for rendering the page
    initial_data = get_data_for_dropdown("Argentina-total", True, 0, 1000)
    return render(
        request,
        "webcovid/index.html",
        {
            "initial_data": str(initial_data).replace("'", '"'),
            "regiones": list(models.nodes.keys()),
        },
    )


def get_data_for_dropdown(region, tendencia, start, end):
    # Logic to get data based on the selected dropdown value
    # Replace this with your actual data fetching logic
    result = models.mostrar_curva2(region, tendencia, start, end)
    return result


def get_data(request):
    # Endpoint for AJAX data request
    data = request.POST
    node = data.get("region", "Argentina")
    tendencia = data.get("tendencia", True)
    inicio = int(data.get("start", 0))
    fin = int(data.get("end", 1000))
    data = get_data_for_dropdown(node, tendencia, inicio, fin)
    return JsonResponse(data)
