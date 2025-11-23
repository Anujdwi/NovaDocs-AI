# PowerShell script to test the chat endpoint

$uri = "http://localhost:8000/api/v1/chat/"
$body = @{
    message = "What is this document about?"
    # session_id = 1  # optional - uncomment if you have a session
} | ConvertTo-Json

Write-Host "Sending request to: $uri"
Write-Host "Body: $body"
Write-Host ""

try {
    $response = Invoke-RestMethod -Uri $uri -Method Post -Body $body -ContentType "application/json"
    Write-Host "`n✓ SUCCESS - Response:" -ForegroundColor Green
    $response | ConvertTo-Json -Depth 10
} catch {
    Write-Host "`n✗ ERROR:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Yellow
    
    if ($_.Exception.Response) {
        $statusCode = $_.Exception.Response.StatusCode.value__
        Write-Host "`nStatus Code: $statusCode" -ForegroundColor Yellow
        
        try {
            $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
            $responseBody = $reader.ReadToEnd()
            $reader.Close()
            
            Write-Host "`nResponse Body:" -ForegroundColor Yellow
            # Try to parse as JSON for better formatting
            try {
                $jsonResponse = $responseBody | ConvertFrom-Json
                $jsonResponse | ConvertTo-Json -Depth 10
            } catch {
                Write-Host $responseBody
            }
        } catch {
            Write-Host "Could not read response body: $_"
        }
    } else {
        Write-Host "`nNote: Server might not be running. Check:" -ForegroundColor Cyan
        Write-Host "  1. Is Django server running? (docker-compose up or python manage.py runserver)" -ForegroundColor Cyan
        Write-Host "  2. Is the server accessible at $uri" -ForegroundColor Cyan
    }
}

