// Binary xla-preinstall triggers XLA PJRT CPU plugin download during Docker build.
// This ensures runtime containers have the plugin pre-cached and need no network
// access or writable HOME directory.
package main

import (
	"fmt"
	"os"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
)

func main() {
	b := backends.MustNew()
	b.Finalize()
	fmt.Fprintf(os.Stderr, "XLA PJRT plugin installed successfully\n")
}
