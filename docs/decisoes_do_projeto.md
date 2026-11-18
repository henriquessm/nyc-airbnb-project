# Guia para a prova: lógica de negócio e decisões do modelo

## 1. Qual problema estamos resolvendo?

Queremos estimar o **preço anunciado por noite** de um Airbnb em Nova York a partir das características do anúncio. É regressão supervisionada: aprendemos com exemplos que já têm preço e prevemos um valor contínuo.

A utilidade de negócio é oferecer uma referência para comparar anúncios semelhantes e apoiar uma decisão de preço. **O modelo não encontra o preço que maximiza receita**: não temos reservas efetivas, elasticidade da demanda, custos ou ocupação observada. Preço anunciado também não comprova preço pago.

O conjunto é um retrato de 2019. A avaliação demonstra desempenho nesse universo histórico, não nos preços atuais ou em outras cidades.

## 2. Por que o preço exige um tratamento especial?

A maioria dos anúncios tem preço relativamente baixo, mas poucos são muito caros. A média é US$ 152,76, a mediana é US$ 106 e o máximo é US$ 10.000. Essa diferença mostra uma distribuição com cauda longa à direita.

Se treinássemos com erro quadrático diretamente em dólares, poucos anúncios caríssimos poderiam dominar o aprendizado. A rede gastaria muita capacidade tentando acertar esses casos, embora eles representem uma parcela pequena dos anúncios.

**Resposta para a prova:** “O mercado é muito assimétrico. Precisamos estimar bem o preço dos anúncios comuns sem deixar que poucos extremos controlem todo o treinamento.”

## 3. Por que usamos log1p no alvo?

Transformamos o alvo em `z = log(1 + price)`. Isso comprime os preços altos e faz o treinamento dar mais atenção a diferenças proporcionais. Errar aproximadamente de 100 para 200 e de 1.000 para 2.000 tem impacto parecido no espaço log, embora o segundo erro em dólares seja muito maior.

A assimetria observada cai de 19,119 para 0,629. Isso ajuda a trabalhar com uma escala mais equilibrada, mas **a rede não exige que o alvo tenha distribuição normal**.

O `+1` permite lidar com zero e mantém consistência com as transformações dos atributos. Como preços zero são removidos, essa propriedade não é indispensável para o alvo deste arquivo.

Para voltar a dólares usamos `expm1(z) = exp(z) - 1`. Há uma consequência: inverter a média prevista do log não equivale a prever a média do preço em dólares. O modelo pode se aproximar de um preço típico e subestimar a cauda cara. Não há correção desse efeito implementada.

**Resposta para a prova:** “Usamos log1p para reduzir a influência dos extremos e aprender diferenças relativas de preço. Depois voltamos a dólares para interpretar o resultado. O custo é que podemos perder precisão monetária nos anúncios muito caros.”

## 4. Qual é a loss e por que ela foi escolhida?

A rede usa **MSE no log-preço**: `média((log_previsto - log1p(preço_real))²)`. Ela é simples, diferenciável e penaliza mais os erros grandes na escala em que treinamos. A transformação log reduz o domínio dos extremos antes de aplicar a penalização quadrática.

Essa escolha é uma referência prática de regressão neural; não houve experimento demonstrando que ela vence todas as alternativas. MSE em log continua sensível a erros grandes nessa escala.

MAE seria uma alternativa menos sensível a extremos; Huber combina comportamento quadrático para erros pequenos e linear para erros grandes. Nenhuma foi comparada neste projeto.

**Ponto essencial:** minimizar MSE em log não é o mesmo que minimizar MAE em dólares. A loss orienta os gradientes; a métrica de negócio orienta a comparação dos modelos.

## 5. Por que a métrica principal é MAE em dólares?

O MAE é a média dos erros absolutos: `média(abs(preço_previsto - preço_real))`. Sua vantagem de negócio é ser comunicado diretamente na unidade da decisão: dólares por diária.

MAE de aproximadamente US$ 60 significa erro absoluto médio de US$ 60 no conjunto avaliado. Não significa que toda previsão erre esse valor, nem que haja perda média de receita de US$ 60: não medimos o efeito do preço sobre reservas.

Cada anúncio recebe o mesmo peso. O MAE não dá um peso quadrático aos erros extremos, embora erros monetários grandes ainda contribuam bastante para a média. Não modelamos custos diferentes para superestimar ou subestimar.

Usamos também **RMSE**, que evidencia falhas muito grandes, e **R²**, que compara o erro quadrático com prever a média do conjunto avaliado. R² não é porcentagem de acertos e pode ser negativo.

**Resposta para a prova:** “MAE em dólares é fácil de interpretar para quem decide preços. RMSE mostra o impacto dos erros extremos e R² fornece uma referência complementar de ajuste.”

## 6. Por que precisamos de baselines?

Uma rede mais complexa só faz sentido se melhorar uma solução simples. Baselines respondem se os atributos ajudam e se a não linearidade da rede gera ganho suficiente para justificar seu custo.

O **DummyRegressor** prevê uma constante: a média do log-preço do treino, convertida de volta para dólares. É uma referência sem uso das características do anúncio. Essa constante não é a média aritmética dos preços. Uma baseline pela mediana em dólares também seria útil para MAE, mas não foi implementada.

A **Ridge** é uma regressão linear regularizada no log-preço, usando os mesmos atributos transformados. Ela testa quanto conseguimos explicar com efeitos aditivos simples, com treinamento rápido e menor complexidade operacional. A penalização L2 estabiliza coeficientes diante de variáveis relacionadas. Após a inversão do log, efeitos aditivos se tornam aproximadamente multiplicativos no preço.

O alpha foi escolhido na validação entre `0.01, 0.1, 1, 10, 100`; venceu `10`.

**Resposta para a prova:** “O dummy testa se aprendemos algo além de um preço constante; a Ridge testa se precisamos mesmo de uma rede para superar uma solução linear barata.”

## 7. Por que uma MLP?

Os dados são tabulares, e uma MLP recebe diretamente um vetor de características. Suas camadas e ativações não lineares permitem aprender interações: por exemplo, uma condição de estadia pode se relacionar com preço de maneira diferente conforme localização e tipo de quarto.

O contexto da disciplina também motiva estudar redes neurais. Isso não demonstra que redes sejam a melhor família para esta tabela. Modelos de árvores com boosting seriam comparações úteis, mas não foram testados.

A rede começa com duas camadas ocultas de 256 e 128 unidades. A busca compara tamanhos menores, maiores e mais profundos. A configuração escolhida tem **128 e 64 unidades**, 119 entradas e uma saída, totalizando **23.681 parâmetros**.

Usamos **ReLU** por ser simples, barata e permitir não linearidade. A inicialização **He** é adequada à ReLU e ajuda a preservar a escala do sinal pelas camadas. A saída é linear porque prevemos um valor contínuo no espaço log. Ela não impõe positividade: o código não aplica piso ao preço convertido.

**Resposta para a prova:** “A MLP permite aprender interações que a Ridge não representa diretamente. Escolhemos uma rede menor porque aumentar a capacidade não trouxe ganho relevante na busca realizada.”

## 8. Por que AdamW?

AdamW adapta as atualizações de cada parâmetro usando o histórico dos gradientes e de suas magnitudes. É uma escolha prática para treinar uma MLP com entradas heterogêneas, sem precisar usar a mesma escala efetiva de atualização em todos os pesos.

Seu diferencial é aplicar **weight decay desacoplado** da atualização adaptativa. Isso facilita experimentar penalização dos pesos separadamente da dinâmica do Adam.

**Atenção para a prova:** o modelo final usa AdamW com `weight_decay=0`. Portanto, ele não recebe regularização por weight decay. Usar AdamW não significa que essa penalização esteja necessariamente ativa. Também não houve comparação com SGD; não podemos afirmar que AdamW foi comprovadamente o melhor otimizador.

A taxa final é `1e-3`. Foram comparadas `3e-4` e `3e-3`. O batch de 256 equilibra custo e frequência de atualização, mas seu tamanho não foi otimizado em uma busca própria.

## 9. Por que separar treino, validação e teste?

O **treino** ajusta os pesos. A **validação** escolhe configurações e o momento de parar. O **teste** estima o desempenho após essas escolhas. Se escolhêssemos o modelo pelo teste, estaríamos adaptando a solução à própria prova final e superestimando sua qualidade.

Usamos aproximadamente 68% para treino, 12% para validação e 20% para teste: 33.240, 5.867 e 9.777 linhas. Primeiro reservamos 20% para teste; depois retiramos 15% dos 80% restantes para validação.

A divisão é estratificada por decis do preço para preservar proporções de faixas de preço. O código usa decis de `log1p(price)`, mas, como log preserva a ordem, isso é essencialmente equivalente a usar decis do preço original.

O pipeline aprende escalas, quantis e categorias somente no treino. Validação e teste recebem essas mesmas regras sem reaprendê-las.

**Limitação:** a divisão é aleatória por anúncio. O mesmo anfitrião pode aparecer nos dois lados. Isso avalia anúncios do mesmo universo, mas não comprova generalização para anfitriões inéditos ou períodos futuros. A inspeção inicial da EDA também examinou o arquivo completo; o teste não ficou absolutamente invisível em toda a exploração.

## 10. Como evitamos overfitting?

Overfitting ocorre quando melhorar no treino deixa de melhorar em dados separados. Por isso monitoramos a validação e usamos **early stopping**: até 300 épocas, interrompendo após 20 épocas sem redução da MSE de validação no log.

O código restaura os pesos da melhor época, não os da última. Na seed 42 do modelo final, a melhor época foi 43 e a execução parou na 63.

Também testamos dropout de 0,2 e weight decay de 0,05. Eles não melhoraram o MAE de validação nas configurações avaliadas. O modelo final tem ambos iguais a zero, mas continua regularizado pelo early stopping e pela menor capacidade.

Dropout reduziu a diferença entre perdas de treino e validação, sem melhorar a métrica principal. **Gap menor sozinho não garante modelo melhor**: pode apenas indicar pior ajuste ao treino.

## 11. Como escolhemos o modelo final?

A busca avalia oito configurações, alterando um fator por vez: tamanho, profundidade, taxa de aprendizado, dropout ou weight decay. Isso facilita interpretar os efeitos com custo limitado, mas não explora todas as combinações.

A regra é menor **MAE na validação em dólares**. Dentro de cada treinamento, a época é escolhida por **MSE em log**. São critérios relacionados, mas não idênticos: a melhor época em uma métrica pode não ser a melhor na outra.

A rede menor e a base com taxa `3e-4` chegaram a US$ 59,23 de MAE de validação. O código arredonda a métrica antes de ordenar; o empate mantém a rede menor, inserida antes. Menor custo justifica sua preferência, mas não devemos dizer que houve superioridade clara em precisão.

Mais capacidade não ajudou muito no espaço pesquisado. Isso sugere priorizar qualidade dos dados e novas características, sem provar que toda melhoria de arquitetura seja impossível.

## 12. A rede valeu a pena para o negócio?

| Modelo | MAE no teste |
|---|---:|
| Dummy | US$ 86,06 |
| Ridge | US$ 62,18 |
| MLP, média de três seeds | US$ 60,20 ± US$ 0,33 |

A rede reduziu o MAE da Ridge em **3,19%**, aproximadamente US$ 1,98 de erro absoluto médio por anúncio. O projeto exigia **10%** de melhoria para justificar a complexidade, o que corresponderia a MAE de até US$ 55,96. **O critério não foi atingido.**

O limiar de 10% é uma regra de aceitação do experimento; não foi derivado de um cálculo de lucro. A conclusão equilibrada é que a rede melhora um pouco a previsão, mas o ganho observado não atende à exigência definida para justificar seu custo adicional.

As seeds 42, 43 e 44 verificam estabilidade do treinamento no mesmo split. A média apresentada é a média das métricas, não a previsão de um ensemble. O desvio entre seeds não mede toda a incerteza da avaliação nem prova significância estatística.

**Resposta para a prova:** “Houve ganho preditivo, mas insuficiente pelo critério do projeto. Uma solução mais complexa não se justifica apenas por ganhar alguns centavos ou dólares na métrica; precisamos comparar benefício e custo.”

## 13. Onde o modelo erra e qual a consequência?

O erro aumenta nos anúncios caros. No grupo com preço de pelo menos US$ 500, o MAE do modelo seed 42 é US$ 858,12 e há forte subestimação. Abaixo de US$ 50, a superestimação média é US$ 17,91.

Isso é compatível com uma previsão próxima do preço típico e com falta de atributos como tamanho, acabamento e amenidades. A transformação log também reduz a influência monetária dos extremos. São explicações plausíveis, não causas isoladas comprovadas por ablação.

Superestimar pode tornar um anúncio pouco competitivo; subestimar pode levar a cobrar menos do que seria possível. O projeto não mediu essas perdas efetivamente. Por isso, uma previsão pontual não deveria ser confundida com uma recomendação comercial garantida.

A análise por região e quarto evita que a média global esconda problemas de segmentos. Mostrar a quantidade de exemplos importa: um resultado de um grupo pequeno é menos estável. O indicador `MAE/mediana do preço` contextualiza o erro do grupo, mas não é MAPE.

Os maiores erros levantam hipóteses de luxo, espaços para eventos ou preços mensais/simbólicos. Não é correto declarar esses casos inválidos sem confirmação, nem dividir automaticamente por 30 um preço apenas porque a estadia mínima é longa.

## 14. Como explicamos o que a rede aprendeu?

Usamos **permutation importance**: embaralhamos uma característica e medimos quanto o MAE piora. Quanto maior a piora, mais o modelo depende daquela informação no conjunto avaliado.

O embaralhamento ocorre na coluna original, antes do pipeline, para que todos os indicadores de um bairro sejam avaliados juntos. Cinco repetições ajudam a observar a variabilidade desse diagnóstico.

`room_type` foi a informação mais importante, seguida por sinais de localização e operação. Uma região com importância pequena pode estar representada também por bairro e coordenadas. **Importância preditiva não é causalidade**: não significa que alterar uma característica causará o efeito observado no preço.

## 15. EDA: o essencial para uma pergunta

- **Alvo assimétrico:** média acima da mediana e poucos preços extremos; motivou o log.
- **Tipo de quarto importa:** medianas de US$ 160 para imóvel inteiro, US$ 70 para quarto privado e US$ 45 para compartilhado.
- **Localização importa:** mapas e comparações por bairros mostram gradientes e diferenças de preço.
- **Categorias desbalanceadas:** alguns bairros têm poucos exemplos, dificultando estimativas estáveis.
- **Contagens têm cauda longa:** motivaram clipping, log e padronização das entradas.
- **Ausência pode ter significado:** `reviews_per_month` ausente coincide com ausência de avaliações; foi preenchido com zero.
- **Correlação não explica tudo:** associações marginais fracas podem esconder interações aprendidas pela rede.
- **PCA, t-SNE e UMAP foram exploratórios:** ajudaram a visualizar estrutura, mas não entraram como atributos da rede. As duas primeiras componentes da PCA explicaram 44,9% da variância das 15 entradas usadas nessa análise.

Os filtros e transformações buscam preservar uma população útil e tornar o treinamento estável. Preços zero foram removidos; preços altos positivos foram mantidos para não excluir casos difíceis sem evidência. Identificadores e texto ficaram fora da modelagem tabular simples, ao custo de perder possíveis sinais úteis.

## 16. Melhorias que fazem sentido defender

Investigar a qualidade do alvo com critérios verificáveis; comparar Huber e MAE; testar árvores com boosting; adicionar características de tamanho e qualidade; avaliar por anfitrião e por período; e estimar intervalos de previsão. Essas propostas atacam limitações observadas, mas ainda precisam de experimento.

A principal ideia para defender é: **a complexidade deve responder a uma necessidade e produzir ganho mensurável**. O projeto mostrou que atributos simples já explicam bastante com Ridge e que aumentar a rede trouxe pouco ganho. O próximo passo deve buscar informação melhor e uma avaliação mais próxima do cenário de uso.

## 17. Mapa rápido do código para consulta

| Arquivo | O que procurar |
|---|---|
| `code/preprocess.py` | Divisão estratificada; transformações aprendidas só no treino; log do alvo. |
| `code/model.py` | Arquitetura; MSE em log; AdamW; batches; early stopping e restauração dos pesos. |
| `code/regression.py` | Baselines; escolha pelo MAE; comparação de configurações; teste; análise de erros. |
| `code/eda.py` | Distribuições, diferenças entre grupos, correlações e projeções exploratórias. |

Fonte dos valores: `docs/projects/eda/results.json` e `docs/projects/regression/results.json`. As justificativas explicam a lógica técnica e comercial da implementação; escolhas heurísticas não devem ser apresentadas como ótimos comprovados.
