import math
from .models import Paper
from .models import RecentPaper
from django.contrib import admin
from django.db import transaction


admin.register(Paper)
admin.site.register(RecentPaper)
