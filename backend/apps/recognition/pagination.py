"""分页配置：允许前端通过 ``page_size`` 参数调整每页数量。"""

from rest_framework.pagination import PageNumberPagination


class StandardPagination(PageNumberPagination):
    page_size = 10
    page_size_query_param = "page_size"
    max_page_size = 200
