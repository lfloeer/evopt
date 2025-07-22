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
	"github.com/olekukonko/tablewriter/tw"
	"github.com/samber/lo"
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

	tw := tablewriter.WithConfig(tablewriter.Config{
		Row: tw.CellConfig{
			Alignment: tw.CellAlignment{Global: tw.AlignRight},
		},
	})

	{
		table := tablewriter.NewTable(os.Stdout, tw)
		headers := []string{"Hour", "Forecast", "TotalDemand", "GridImportCost", "GridExportCost"}

		for i, goal := range *req.TimeSeries.BGoal {
			if lo.Sum(goal) > 0 {
				headers = append(headers,
					fmt.Sprintf("Bat %d Goal", i),
				)
			}
		}

		table.Header(headers)

		for t := range len(req.TimeSeries.Ft) {
			row := []string{
				strconv.Itoa(t),
				strconv.Itoa(int((req.TimeSeries.Ft)[t])),
				strconv.Itoa(int((req.TimeSeries.Gt)[t])),
				str2((req.TimeSeries.PN)[t]),
				str2((req.TimeSeries.PE)[t]),
			}

			for _, goal := range *req.TimeSeries.BGoal {
				if lo.Sum(goal) > 0 {
					row = append(row, str(goal[t]))
				}
			}

			table.Append(row)
		}

		table.Render()
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

	{
		table := tablewriter.NewTable(os.Stdout, tw)
		headers := []string{
			"Hour", "Forecast",
			// "FlowDirection",
			"GridImport", "GridExport",
		}

		for i := range *res.Batteries {
			headers = append(headers,
				fmt.Sprintf("Bat %d Cha", i), // ChargingPower
				fmt.Sprintf("Bat %d Dis", i), // DischargingPower
				fmt.Sprintf("Bat %d Soc", i),
			)
		}

		table.Header(headers)

		for t := range len(*res.FlowDirection) {
			row := []string{
				strconv.Itoa(t),
				str((req.TimeSeries.Ft)[t]),
				// str((*res.FlowDirection)[t]),
				str((*res.GridImport)[t]),
				str((*res.GridExport)[t]),
			}

			for j, b := range *res.Batteries {
				_ = j
				row = append(row,
					str((*b.ChargingPower)[t]),
					str((*b.DischargingPower)[t]),
					str((*b.StateOfCharge)[t]),
					// str((*b.StateOfCharge)[i]/req.Batteries[j].SMax*100),
				)
			}

			table.Append(row)
		}

		table.Render()
	}
}

func str(f float32) string {
	if f == 0 {
		return "-"
	}
	return fmt.Sprintf("%.0f", f)
}

func str2(f float32) string {
	if f == 0 {
		return "-"
	}
	return fmt.Sprintf("%.2f", f)
}
