package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/andig/evopt/client"
	"github.com/olekukonko/tablewriter"
)

func main() {
	// custom HTTP client
	hc := http.Client{}

	c, err := client.NewClientWithResponses("http://localhost:7050", client.WithHTTPClient(&hc))
	if err != nil {
		log.Fatal(err)
	}

	example, err := c.GetOptimizeExampleWithResponse(context.TODO())
	if err != nil {
		log.Fatal(err)
	}

	if example.StatusCode() != http.StatusOK {
		log.Fatalf("Expected HTTP 200 but received %d\n%s", example.StatusCode(), string(example.Body))
	}

	req := *example.JSON200
	{
		b, _ := json.MarshalIndent(req, "", "  ")
		fmt.Println(string(b))
	}

	resp, err := c.PostOptimizeChargeScheduleWithResponse(context.TODO(), req)
	if err != nil {
		log.Fatal(err)
	}

	if resp.StatusCode() == http.StatusInternalServerError && resp.JSON500.Message != nil {
		log.Fatalf("Expected HTTP 200 but received %d\n%s", resp.StatusCode(), *resp.JSON500.Message)
	}

	if resp.StatusCode() != http.StatusOK {
		log.Fatalf("Expected HTTP 200 but received %d\n%s", resp.StatusCode(), string(resp.Body))
	}

	res := *resp.JSON200
	{
		b, _ := json.MarshalIndent(res, "", "  ")
		fmt.Println(string(b))
	}

	table := tablewriter.NewTable(os.Stdout)
	headers := []string{"Hour", "Forecast", "FlowDirection", "GridImport", "GridExport"}

	for i := range *res.Batteries {
		headers = append(headers, []string{
			fmt.Sprintf("Bat %d ChargingPower", i),
			fmt.Sprintf("Bat %d DischargingPower", i),
			fmt.Sprintf("Bat %d SOC", i),
		}...)
	}

	table.Header(headers)

	for i := range len(*res.FlowDirection) {
		row := []string{
			strconv.Itoa(i),
			strconv.Itoa(int((req.TimeSeries.Ft)[i])),
			strconv.Itoa(int((*res.FlowDirection)[i])),
			strconv.Itoa(int((*res.GridImport)[i])),
			strconv.Itoa(int((*res.GridExport)[i])),
		}

		for _, b := range *res.Batteries {
			row = append(row, []string{
				str((*b.ChargingPower)[i]),
				str((*b.DischargingPower)[i]),
				str((*b.StateOfCharge)[i]),
			}...)
		}

		table.Append(row)
	}

	table.Render()
}

func str(f float32) string {
	if f == 0 {
		return "-"
	}
	return fmt.Sprintf("%.2f", f)
}
