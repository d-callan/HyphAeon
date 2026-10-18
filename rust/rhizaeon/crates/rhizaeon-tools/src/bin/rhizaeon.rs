#[path = "../../../../../../rhizaeon/native/src/cli.rs"]
mod cli;

fn main() {
    let code = cli::cli_main(std::env::args().collect());
    std::process::exit(code);
}
