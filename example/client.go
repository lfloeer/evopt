package main

import (
	"context"
	"log"
	"math"
	"net/http"

	"github.com/andig/evopt/client"
	"github.com/samber/lo"
)

func main() {
	// custom HTTP client
	hc := http.Client{}

	// with a raw http.Response
	{
		c, err := client.NewClientWithResponses("http://localhost:7050", client.WithHTTPClient(&hc))
		if err != nil {
			log.Fatal(err)
		}

		data := client.OptimizationInput{
			M: lo.ToPtr[float32](math.MaxFloat32),
		}

		resp, err := c.PostOptimizeChargeScheduleWithResponse(context.TODO(), data)
		if err != nil {
			log.Fatal(err)
		}

		if resp.StatusCode() == http.StatusInternalServerError && resp.JSON500.Message != nil {
			log.Fatalf("Expected HTTP 200 but received %d\n%s", resp.StatusCode(), *resp.JSON500.Message)
		}

		if resp.StatusCode() != http.StatusOK {
			log.Fatalf("Expected HTTP 200 but received %d\n%s", resp.StatusCode(), string(resp.Body))
		}
	}
}
