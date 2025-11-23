# Quick health check script
$uri = "http://localhost:8000/api/health/"

Write-Host "Checking server health at: $uri" -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod -Uri $uri -Method Get
    Write-Host "✓ Server is running!" -ForegroundColor Green
    $response | ConvertTo-Json
} catch {
    Write-Host "✗ Server is not responding" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Yellow
    Write-Host "`nMake sure the server is running:" -ForegroundColor Cyan
    Write-Host "  - Docker: docker-compose up" -ForegroundColor Cyan
    Write-Host "  - Local: python manage.py runserver" -ForegroundColor Cyan
}

