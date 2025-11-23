from rest_framework import serializers

class IngestSerializer(serializers.Serializer):
    title = serializers.CharField()
    file = serializers.FileField(required=False)
    source_url = serializers.URLField(required=False)