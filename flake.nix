{
  description = "A Python 3.11 environment.";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        system = system;
      };
      en = pkgs.python311Packages.buildPythonPackage {
        pname = "en_core_web_sm";
        version = "3.7.1";
        src = pkgs.fetchurl {
          url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz";
          hash = "sha256-FKLzG8R2r4cBmBnqjJlI+r39RzpELt1rHLpivwwsD1U=";
        };
        buildInputs = with pkgs.python311Packages; [ pipBuildHook ];
        dependencies = with pkgs.python311Packages; [ spacy ];
      };
    in
    {
      devShell.${system} = with pkgs; mkShell {
        buildInputs = [
          (python311.withPackages (ps: with ps; [
            en
            gensim
            pip
            polars
            spacy
            tqdm
            zstandard
          ]))
        ];
      };
    };
}
